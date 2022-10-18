import torch
from math import cos, pi


class LRScheduler:
    def __init__(self, args, optimizer, start_epoch, dataset):
        self.optimizer = optimizer
        if args.scheduler == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=args.lr_decay, last_epoch=start_epoch)
        elif args.scheduler == "exp":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=float(args.lr_decay[0]), last_epoch=start_epoch)
        elif args.scheduler == "cosine_anneal_warmup":
            warmup = 5 if dataset == "imagenet" else 10
            self.lr_scheduler = CosineAnnealingWarmup(optimizer, last_epoch=start_epoch, warmup_epoch=warmup,
                                                      max_epoch=args.epochs)
            self.lr_scheduler.step()
        else:
            raise NotImplementedError

    def step(self):
        self.lr_scheduler.step()

    def getLR(self):
        return self.optimizer.param_groups[0]['lr']


class CosineAnnealingWarmup:
    def __init__(self, optimizer, last_epoch=-1, warmup_epoch=10, max_epoch=100):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch
        self.lr_max = self.optimizer.param_groups[0]['lr']
        self.lr_min = self.lr_max * 0.001
        self.max_epoch = max_epoch

    def step(self):
        self.last_epoch += 1
        self.update()

    def update(self):
        if self.last_epoch < self.warmup_epoch:
            lr = (self.lr_max - self.lr_min) * (self.last_epoch + 1) / self.warmup_epoch
        else:
            lr = self.lr_min + (self.lr_max - self.lr_min) * (
                    1 + cos(pi * (self.last_epoch - self.warmup_epoch) / (self.max_epoch - self.warmup_epoch))) / 2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
