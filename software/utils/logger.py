import torch
import utils.utils as utils
import os
import sys


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def get_metrics(num):
    return [AverageMeter() for _ in range(num)]


def layer_count(args):
    if args.model == "resnet50":
        layer_cnt = 16
    elif args.model == "resnet101":
        layer_cnt = 36
    elif args.model == "MobileNetV2":
        layer_cnt = 17
    elif args.model == "resnet32":
        layer_cnt = 15
    elif args.model == "MobileNetV2_32x32":
        layer_cnt = 12
    else:
        layer_cnt = 20
    return layer_cnt


def record_msg(msg_path, msgs, epoch=None):
    if msg_path is not None:
        with open(msg_path, "a+") as f:
            if epoch is not None:
                f.write("Epoch {}\n".format(epoch))
            for msg in msgs:
                f.write(msg)
            f.flush()
            os.fsync(f)


def handle_loggers(args, model=None, global_file=False):
    train_log, test_log, msg_log, global_log = None, None, None, \
                                               os.path.join(os.path.join("/".join(args.save_dir.split("/")[:-1])), "group_result.txt")

    if not args.evaluate and len(args.save_dir) > 0:
        if not os.path.exists(os.path.join(args.save_dir)):
            os.makedirs(os.path.join(args.save_dir))

        if global_file:
            if not os.path.exists(global_log):
                with open(global_log, "w") as f:
                    f.write("Name\ttop1\tMMac\ttop5\ttask-loss\tsparsity-loss\n")
        else:
            global_log = None

        if model and not os.path.exists(os.path.join(args.save_dir, "model.log")):
            with open(os.path.join(args.save_dir, "model.log"), "w") as f:
                print(model, file=f)

        config_log, train_log, test_log, msg_log = os.path.join(args.save_dir, "config.log"), \
                                                   os.path.join(args.save_dir, "train.log"), \
                                                   os.path.join(args.save_dir, "test.log"), \
                                                   os.path.join(args.save_dir, "message.log"),

        cmd = utils.generate_cmd(sys.argv[1:])
        test_cmd = utils.generate_test_cmd(cmd[:-1], args)
        if not (os.path.exists(config_log) and not args.auto_resume):
            with open(config_log, "w") as f:
                f.write(cmd + "\n\n")
                f.write(test_cmd + "\n\n")
                print('Args:', args, file=f)
                f.write("\n")
                for k, v in vars(args).items():
                    f.write("{k} : {v}\n".format(k=k, v=v))
                f.flush()
                os.fsync(f)

        if not os.path.exists(test_log):
            with open(test_log, "w") as f:
                f.write("epoch\ttop1\ttop5\ttask-loss\tspatial-loss\tchannel-loss\tnet-loss\tMMac\t\n")
        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("epoch\tlr\tgumbel-temp\tgumbel-noise\ttop1\ttop5\ttask-loss\tspatial-loss\tchannel_loss\t"
                        "net-loss\t\n")
    elif args.evaluate:
        if global_file:
            try:
                if not os.path.exists(global_log):
                    with open(global_log, "w") as f:
                        f.write("Name\ttop1\tMMac\ttop5\ttask-loss\tsparsity-loss\n")
            except:
                global_log = None
        else:
            global_log = None
    if args.save_dir:
        os.makedirs(os.path.join(args.save_dir, "analysis"), exist_ok=True)
    return train_log, test_log, msg_log, global_log


class MetricRecorder:
    def __init__(self, args, phase):
        self.phase = phase
        self.top1, self.top5, self.task_loss_record, self.spatial_loss_record, self.channel_loss_record, \
            self.net_loss_record, self.MMac_record = get_metrics(7)
        layer_cnt = layer_count(args)

        self.spatial_sparsity_records = [AverageMeter() for _ in range(layer_cnt)]
        self.channel_sparsity_records = [AverageMeter() for _ in range(layer_cnt)]

    def update(self, metrics, list_metrics, inp):
        prec1, prec5, t_loss, s_loss, c_loss, net_loss, flops = metrics
        self.top1.update(prec1.item(), inp.size(0))
        self.top5.update(prec5.item(), inp.size(0))
        self.task_loss_record.update(t_loss.item(), inp.size(0))
        self.spatial_loss_record.update(s_loss.item(), inp.size(0))
        self.channel_loss_record.update(c_loss.item(), inp.size(0))
        self.net_loss_record.update(net_loss.item(), inp.size(0))
        self.MMac_record.update(sum(flops)/len(flops), len(flops))
        s_percents, c_percents = list_metrics

        for s_per, recorder in zip(s_percents, self.spatial_sparsity_records):
            recorder.update(s_per.item(), 1)
        for c_per, recorder in zip(c_percents, self.channel_sparsity_records):
            recorder.update(c_per.item(), 1)

    def summary(self, epoch, msg_path="", tb="", logger_path="", model=None, metas=None, print_info=True):
        top1, top5, task_loss, spatial_loss, channel_loss, net_loss\
            = self.top1.avg, self.top5.avg, self.task_loss_record.avg, self.spatial_loss_record.avg, \
              self.channel_loss_record.avg, self.net_loss_record.avg
        # MMac = model.compute_average_flops_cost()[0] / 1e6 if model is not None else 0
        MMac = self.MMac_record.avg / 1e6
        spatial_layer_str = ",".join([str(round(recorder.avg, 2)) for recorder in self.spatial_sparsity_records])
        channel_layer_str = ",".join([str(round(recorder.avg, 2)) for recorder in self.channel_sparsity_records])
        if print_info:
            print(f'* Epoch {epoch} - Prec@1 {top1:.3f} - Prec@5 {top5:.3f}')
            if model:
                print(f'* average FLOPS (multiply-accumulates, MACs) per image: '
                      f'{MMac:.6f} MMac')
            print("* Spatial Percentage are: {}".format(spatial_layer_str))
            print("* Channel Percentage are: {}".format(channel_layer_str))

        if logger_path:
            with open(logger_path, "a+") as f:
                if self.phase == "train":
                    f.write("{epoch}\t{lr:.4f}\t{gumbel_temp}\t{gumbel_noise}\t{top1.avg:.4f}\t{top5.avg:.4f}\t"
                            "{task_loss.avg:.4f}\t{spatial_loss.avg:.4f}\t{channel_loss.avg:.4f}\t"
                            "{net_loss.avg:.4f}\n".format(epoch=epoch, lr=metas[0], gumbel_temp=metas[1], gumbel_noise=metas[2], top1=self.top1,
                        top5=self.top5, task_loss=self.task_loss_record, spatial_loss=self.spatial_loss_record,
                        channel_loss=self.channel_loss_record, MMac=MMac, net_loss=self.net_loss_record))
                else:
                    f.write("{epoch}\t{top1.avg:.4f}\t{top5.avg:.4f}\t{task_loss.avg:.4f}\t{spatial_loss.avg:.4f}\t"
                            "{channel_loss.avg:.4f}\t{net_loss.avg:.4f}\t{MMac:.6f}\n".format(
                        epoch=epoch, top1=self.top1, top5=self.top5, task_loss=self.task_loss_record,
                        spatial_loss=self.spatial_loss_record, channel_loss=self.channel_loss_record, MMac=MMac,
                        net_loss=self.net_loss_record))
                f.flush()
                os.fsync(f)

        msg_strs = [
            "{} Spatial percentage: {}\n".format(self.phase, spatial_layer_str),
            "{} Channel percentage: {}\n".format(self.phase, channel_layer_str)
        ]
        record_msg(msg_path, msg_strs, epoch) if self.phase == "train" else record_msg(msg_path, msg_strs)

        if tb:
            tb.add_scalar("{}/TASK LOSS-EPOCH".format(self.phase), task_loss, epoch)
            tb.add_scalar("{}/Prec@1-EPOCH".format(self.phase), top1, epoch)
            tb.add_scalar('{}/SPATIAL LOSS-EPOCH'.format(self.phase), spatial_loss, epoch)
            tb.add_scalar('{}/CHANNEL LOSS-EPOCH'.format(self.phase), channel_loss, epoch)
            tb.add_scalar('{}/NETWORK LOSS-EPOCH'.format(self.phase), net_loss, epoch)
            if model:
                tb.add_scalar("{}/MMac-EPOCH".format(self.phase), model.compute_average_flops_cost()[0] / 1e6, epoch)
            for idx, recorder in enumerate(self.spatial_sparsity_records):
                tb.add_scalar("{}/SPATIAL LAYER {}-EPOCH".format(self.phase, idx + 1), recorder.avg, epoch)
            for idx, recorder in enumerate(self.channel_sparsity_records):
                tb.add_scalar("{}/CHANNEL LAYER {}-EPOCH".format(self.phase, idx + 1), recorder.avg, epoch)
        return top1, top5, task_loss, spatial_loss, channel_loss, net_loss, MMac


class ErrorAnalyser:
    def __init__(self, path):
        self.file = open(path, "w")
        self.sample_results = [["sample", "target", "pred", "possi", "target_possi", "Mac"]]
        self.sample_cnt = 0

    def update(self, outputs, targets, MMacs, sample_names=None):
        possibs, preds, target_pos = self.extract_meta(outputs, targets)
        if sample_names is None:
            sample_names = [idx+self.sample_cnt for idx in range(len(outputs))]
        else:
            sample_names = list(map(lambda x: x.split("/")[-1], sample_names))
        for sample_name, possib, pred, target, t_pos, MMac in \
                zip(sample_names, possibs, preds, targets, target_pos, MMacs):
            self.sample_results.append(list(map(lambda x:str(x.cpu().tolist()) if isinstance(x, torch.Tensor) else str(x),
                                                [sample_name, target, pred, possib, t_pos, MMac])))
            self.sample_cnt += 1

    def extract_meta(self, outputs, targets):
        pos = torch.softmax(outputs, dim=1)
        preds = torch.max(outputs, dim=1)[1]
        target_pos = []
        for p, target in zip(pos, targets):
            target_pos.append(p[target])
        return torch.max(pos, dim=1)[0], preds, torch.Tensor(target_pos).cuda()

    def finish(self):
        for sample_result in self.sample_results:
            self.file.write(" ".join(sample_result))
            self.file.write("\n")
        self.file.close()
