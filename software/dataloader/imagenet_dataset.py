import os
import torch.utils.data
import torchvision.transforms as transforms
from .imagenet import IN1K


def get_imagenet_dataset(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if not args.evaluate:
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(args.res),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        if len(args.save_dir) > 0:
            if not os.path.exists(os.path.join(args.save_dir)):
                os.makedirs(os.path.join(args.save_dir))
        ## DATA
        trainset = IN1K(root=args.dataset_root, split='train', transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True,
                                                   num_workers=args.workers, pin_memory=False)
    else:
        train_loader = None

    transform_val = transforms.Compose([
        transforms.Resize(int(args.res / 0.875)),
        transforms.CenterCrop(args.res),
        transforms.ToTensor(),
        normalize,
    ])

    valset = IN1K(root=args.dataset_root, split='val', transform=transform_val)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers,
                                             pin_memory=False)
    return train_loader, val_loader
