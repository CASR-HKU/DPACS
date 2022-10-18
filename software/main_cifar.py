from config.args_cifar import args_cifar10 as args
import os.path, sys
import torch
import torch.optim
from dataloader.cifar_dataset import get_cifar_data
from utils.loss import Loss
import tqdm
import utils.flopscounter as flopscounter
import utils.utils as utils
from torch.backends import cudnn as cudnn
import models
from utils.optimizer import select_optimizer
from utils.scheduler import LRScheduler
import utils.logger as logger

try:
    from apex import amp
    mix_precision = False
except:
    mix_precision = False

cudnn.benchmark = True
iteration = 0
device = 'cuda:0'


def main():
    global iteration
    print('Args:', args)

    train_loader, val_loader = get_cifar_data(args)
    ## MODEL
    net_module = models.__dict__[args.model]
    model = net_module(sparse=args.budget >= 0, model_cfg=args.model_cfg, resolution_mask=args.resolution_mask,
                       momentum=args.momentum, budget=args.budget, pooling_method=args.pooling_method,
                       channel_budget=args.channel_budget, group_size=args.group_size,
                       channel_unit_type=args.channel_unit_type, channel_stage=args.channel_stage,
                       before_residual=args.before_residual, full_feature=args.full_feature
                       ).to(device=device)
    meta = {'masks': [], 'device': device, 'gumbel_temp': 5.0, 'gumbel_noise': False, 'epoch': 0,
            "feat_before": [], "feat_after": [], "lasso_sum": torch.zeros(1).cuda(), "channel_prediction": {}}
    _ = model(torch.rand((2, 3, args.res, args.res)).cuda(), meta)

    tb_folder = os.path.join(args.save_dir, "tb") if not args.evaluate else ""
    channel_gumbel = args.channel_budget if "gumbel" in args.channel_unit_type else -1
    criterion = Loss(network_weight=args.net_loss_weight, spatial_weight=args.spatial_loss_weight,
                     num_epochs=args.epochs, channel_weight=args.channel_loss_weight,
                     strategy=args.sparse_strategy, tensorboard_folder=tb_folder, unlimited_lower=args.unlimited_lower,
                     layer_loss_method=args.layer_loss_method, channel_budget=channel_gumbel, spatial_budget=args.budget,
                     network_budget=args.budget, backbone=args.model)
    train_log, test_log, msg_log, global_log = logger.handle_loggers(args, model, global_file=True)

    ## OPTIMIZER
    optimizer = select_optimizer(args, model)

    ## CHECKPOINT
    start_epoch, best_prec1 = utils.load_checkpoint(args, model, optimizer, device)

    scheduler = LRScheduler(args, optimizer, start_epoch, "cifar")
    start_epoch += 1

    best_epoch, best_MMac, best_metric = start_epoch, -1, None

    ## Count number of params
    print("* Number of trainable parameters:", utils.count_parameters(model))

    if mix_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    ## EVALUATION
    if args.evaluate:
        print(f"########## Evaluation ##########")
        validate(args, val_loader, model, criterion, start_epoch)
        return

    ## TRAINING
    for epoch in range(start_epoch, args.epochs):
        print(f"########## Epoch {epoch} ##########")

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(args, train_loader, model, criterion, optimizer, epoch, train_log, msg_log)
        scheduler.step()

        # evaluate on validation set
        metrics = validate(args, val_loader, model, criterion, epoch, test_log, msg_log)

        # remember best prec@1 and save checkpoint
        is_best = metrics[0] > best_prec1
        if is_best:
            best_epoch, best_MMac, best_metric = epoch, metrics[1], metrics
            best_prec1 = max(metrics[0], best_prec1)

        utils.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_prec1': best_prec1,
            "iteration": iteration
        }, folder=args.save_dir, is_best=is_best)
        with open(msg_log, "a+") as f:
            print(f" *Currently Best prec1: {best_prec1}\n-------------------------------------------------\n", file=f)

    with open(msg_log, "a+") as f:
        print(f" * Best prec1: {best_prec1}, Epoch {best_epoch}, MMac {best_MMac}", file=f)


def train(args, train_loader, model, criterion, optimizer, epoch, logger_path=None, msg_path=None):
    """
    Run one train epoch
    """
    global iteration
    model.train()
    Recorder = logger.MetricRecorder(args, "train")
    gumbel_temp, gumbel_noise = utils.set_gumbel(args.gumbel_interval, args.gumbel_temp, epoch/args.epochs,
                                                 args.remove_gumbel)

    num_step = len(train_loader)
    for input, target in tqdm.tqdm(train_loader, total=num_step, ascii=True, mininterval=5):

        input = input.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)

        # compute output
        meta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': gumbel_noise,
                'epoch': epoch, "lasso_sum": torch.zeros(1).cuda(), "channel_prediction": {}}
        output, meta = model(input, meta)

        loss = criterion(output, target, meta)
        iteration = utils.update_tb(criterion, iteration)
        prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))
        Recorder.update([prec1, prec5, criterion.t_loss, criterion.s_loss, criterion.c_loss, criterion.n_loss,
                         criterion.flops],
                        [criterion.percent_spatial, criterion.percent_channel], input)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if mix_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

    metas = [optimizer.param_groups[0]["lr"], gumbel_temp, gumbel_noise]
    Recorder.summary(epoch, msg_path, criterion.tb_writer, logger_path, metas=metas, print_info=False)


def validate(args, val_loader, model, criterion, epoch, logger_path=None, msg_path=None, error_ana_path=None,
             spatial_record_path=None):
    """
    Run evaluation
    """
    Recorder = logger.MetricRecorder(args, "valid")

    # switch to evaluate mode
    model = flopscounter.add_flops_counting_methods(model)
    model.eval().start_flops_count()
    model.reset_flops_count()

    num_step = len(val_loader)
    with torch.no_grad():
        for input, target in tqdm.tqdm(val_loader, total=num_step, ascii=True, mininterval=5):
            input = input.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True)

            # compute output
            meta = {'masks': [], 'device': device, 'gumbel_temp': 1.0, 'gumbel_noise': False, 'epoch': epoch,
                    "feat_before": [], "feat_after": [], "lasso_sum": torch.zeros(1).cuda(), "channel_prediction": {}}
            output, meta = model(input, meta)
            output = output.float()
            loss = criterion(output, target, meta)
            prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))

            Recorder.update([prec1, prec5, criterion.t_loss, criterion.s_loss, criterion.c_loss, criterion.n_loss,
                             criterion.flops.tolist()],
                            [criterion.percent_spatial, criterion.percent_channel], input)

    model.stop_flops_count()
    top1, top5, task_loss, spatial_loss, channel_loss, net_loss, MMac \
        = Recorder.summary(epoch, msg_path, criterion.tb_writer, logger_path, model)

    if args.budget == -1:
        sparsity_loss = 0
    else:
        sparsity_loss = criterion.s_weight * spatial_loss + criterion.c_weight * channel_loss + criterion.n_weight * net_loss
    return top1, MMac, top5, task_loss, sparsity_loss


if __name__ == "__main__":
    main()
