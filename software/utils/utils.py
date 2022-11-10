import os.path
import torch
from torchvision import transforms
from math import cos, pi


def get_inference_time(model, repeat=200, height=416, width=416, device="cuda:0", meta=None, verbose=True):
    import time
    model.eval()
    time_record = []
    with torch.no_grad():
        for i in range(repeat):
            inp = torch.randn(1, 3, height, width)
            start = time.time()
            inp = inp.to(device=device)
            model(inp)
            # model(inp, meta)
            time_record.append(round((time.time() - start), 4))
    avg_infer_time = round(sum(time_record) / len(time_record), 4)
    if verbose:
        print(time_record)
        print("Average runtime: {}".format(avg_infer_time))
    return avg_infer_time


def auto_file_path(checkpoint_path):
    base_dir = "/".join(checkpoint_path.split("/")[:-1])
    return os.path.join(base_dir, "analysis.txt"), os.path.join(base_dir, "spatial.txt")


def update_tb(criterion, iteration, phase="train"):
    if criterion.tb_writer and phase == "train":
        criterion.tb_writer.add_scalar("{}/task loss".format(phase), criterion.t_loss, iteration)
        criterion.tb_writer.add_scalar("{}/network loss".format(phase), criterion.n_loss, iteration)
        criterion.tb_writer.add_scalar("{}/spatial loss".format(phase), criterion.s_loss, iteration)
        criterion.tb_writer.add_scalar("{}/channel loss".format(phase), criterion.c_loss, iteration)
    return iteration + 1



def load_checkpoint(args, model, optimizer, device="cuda:0"):
    global iteration
    start_epoch, best_prec1 = -1, 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            # print('check', checkpoint)
            start_epoch = checkpoint['epoch']-1
            best_prec1 = checkpoint['best_prec1']
            try:
                iteration = checkpoint['iteration']
            except:
                iteration = args.batchsize * start_epoch
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}'' (epoch {checkpoint['epoch']}, best prec1 {checkpoint['best_prec1']})")
        else:
            msg = "=> no checkpoint found at '{}'".format(args.resume)
            if args.evaluate:
                raise ValueError(msg)
            else:
                print(msg)
    elif args.auto_resume and not args.evaluate:
        assert args.save_dir, "Please specify the auto resuming folder"
        resume_path = os.path.join(args.save_dir, "checkpoint.pth")
        if os.path.isfile(resume_path):
            print(f"=> loading checkpoint '{resume_path}'")
            checkpoint = torch.load(resume_path)
            # print('check', checkpoint)
            start_epoch = checkpoint['epoch']-1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{resume_path}'' (epoch {checkpoint['epoch']}, best prec1 {checkpoint['best_prec1']})")
        else:
            msg = "=> no checkpoint found at '{}'".format(resume_path)
            if args.evaluate:
                raise ValueError(msg)
            else:
                print(msg)
    elif args.load:
        try:
            checkpoint_dict = torch.load(args.load, map_location=device)['state_dict']
        except:
            checkpoint_dict = torch.load(args.load, map_location=device)

        model_dict = model.state_dict()
        # update_dict = {k: v for k, v in model_dict.items() if k in checkpoint_dict.keys()}
        update_keys = [k for k, v in model_dict.items() if k in checkpoint_dict.keys()]
        update_dict = {k: v for k, v in checkpoint_dict.items() if k in update_keys}
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)
    return start_epoch, best_prec1


def calculate_baseline(args, module):
    import utils.flopscounter as flopscounter
    model_baseline = module(sparse=False, model_cfg=args.model_cfg, resolution_mask=args.resolution_mask,
                            mask_type=args.mask_type, momentum=args.momentum, budget=-1,
                            mask_kernel=args.mask_kernel, no_attention=args.no_attention,
                            individual_forward=args.individual_forward, save_feat=args.feat_save_dir,
                            target_stage=args.target_stage, mask_thresh=args.mask_thresh,
                            random_mask_stage=args.random_mask_stage, skip_layer_thresh=args.skip_layer_thresh,
                            input_resolution=args.input_resolution, conv1_act=args.conv1_act, group_size=args.group_size,
                            pooling_method=args.pooling_method, channel_budget=args.channel_budget,
                            channel_unit_type=args.channel_unit_type, channel_stage=args.channel_stage,
                            dropout_stages=args.dropout_stages, dropout_ratio=args.dropout_ratio,
                            use_downsample=args.use_downsample, final_activation=args.final_activation,
                            before_residual=args.before_residual).to(device='cuda:0')
    model_baseline = flopscounter.add_flops_counting_methods(model_baseline)
    model_baseline.eval().start_flops_count()
    model_baseline.reset_flops_count()
    meta = {'masks': [], 'device': 'cuda:0', 'gumbel_temp': 5.0, 'gumbel_noise': False, 'epoch': 0,
            "feat_before": [], "feat_after": [], "lasso_sum": torch.zeros(1).cuda(), "channel_prediction": {}}
    _ = model_baseline(torch.rand((2, 3, args.res, args.res)).cuda(), meta)
    model_baseline.stop_flops_count()
    return model_baseline.compute_average_flops_cost()[0] / 1e6


def set_gumbel(intervals, temps, epoch_ratio, remove_gumbel):
    assert len(intervals) == len(temps), "Please reset your gumbel"
    len_gumbel = len(intervals)
    gumbel_temp = temps[-1]
    for idx in range(len(intervals)):
        if intervals[len_gumbel-idx-1] > epoch_ratio:
            gumbel_temp = temps[len_gumbel-idx-1]
        else:
            break
    gumbel_noise = False if epoch_ratio > remove_gumbel else True
    return gumbel_temp, gumbel_noise


def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min=0.0, lr_max=0.1, warmup_epoch=5):
    if current_epoch < warmup_epoch:
        lr = (lr_max-lr_min) * (current_epoch+1) / warmup_epoch
    else:
        lr = lr_min + (lr_max - lr_min) * (
                    1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def tp2str(tp, rounds=None):
    if rounds is None:
        rounds = [4 for _ in range(len(tp))]

    string = ""
    for item, r in zip(tp, rounds):
        string += str(round(float(item), r))
        string += "\t"
    return string + "\n"

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, folder, is_best):
    """
    Save the training model
    """
    if len(folder) == 0:
        print('Did not save model since no save directory specified in args!')
        return
        
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = os.path.join(folder, 'checkpoint.pth')
    print(f" => Saving {filename}")
    torch.save(state, filename)

    if is_best:
        filename = os.path.join(folder, 'checkpoint_best.pth')
        print(f" => Saving {filename}")
        torch.save(state, filename)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.unnormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        assert tensor.shape[0] == 3
        return self.unnormalize(tensor)


def generate_test_cmd(string, args):
    string += " -e --workers 0 --load {}".format(os.path.join(args.save_dir, "checkpoint_best.pth"))
    return string


def generate_cmd(ls):
    string = ""
    for idx, item in enumerate(ls):
        string += item
        string += " "
    return string[:-1] + "\n"
