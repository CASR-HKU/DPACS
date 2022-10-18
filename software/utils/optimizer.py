import torch


def select_optimizer(args, model):
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr,
                                momentum=args.momentum,
                                alpha=0.9,
                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    return optimizer
