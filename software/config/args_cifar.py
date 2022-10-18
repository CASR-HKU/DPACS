import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with sparse masks')

# basic
parser.add_argument('--dataset-root', default='data', type=str, metavar='PATH',
                    help='ImageNet dataset root')
parser.add_argument('--batchsize', default=64, type=int, help='batch size')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--budget', default=-1, type=float,
                    help='computational budget (between 0 and 1) (-1 for no sparsity)')
parser.add_argument('--workers', default=8, type=int, help='number of dataloader workers')
parser.add_argument('--res', default=32, type=int, help='number of epochs')

# learning strategy
parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
parser.add_argument('--lr_decay', default=[30, 60, 90], nargs='+', type=int, help='learning rate decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--optim', type=str, default='sgd', help='Training optimizer')
parser.add_argument('--scheduler', type=str, default='step', help='Training scheduler')

# loss
parser.add_argument('--net_loss_weight', default=10, type=float, help='weight of network sparsity')
parser.add_argument('--spatial_loss_weight', default=10, type=float, help='weight of layer sparsity')
parser.add_argument('--sparse_strategy', type=str, default='static', help='Type of mask')
parser.add_argument('--layer_loss_method', type=str, default='flops', help='Calculation for layer-wise methods')
parser.add_argument('--unlimited_lower', action='store_true', help='loss without lower constraints')

# channel arguments
parser.add_argument('--group_size', type=int, default=1, help='The number for grouping channel pruning')
parser.add_argument('--pooling_method', type=str, default='max', help='Maxpooling or AveragePooling')
parser.add_argument('--channel_budget', default=-1, type=float,
                    help='computational budget (between 0 and 1) (-1 for no sparsity)')
parser.add_argument('--channel_unit_type', type=str, default='fc', help='Type of mask')
parser.add_argument('--channel_stage', nargs="+", type=int, help='target stage for channel masks')
parser.add_argument('--channel_loss_weight', type=float, default=1e-8)
parser.add_argument('--before_residual', action='store_true', help='Use the masked feature for channel pruning before residual')
parser.add_argument('--full_feature', action='store_true', help='Use full feature for channel decision')
parser.add_argument('--dual_fc', action='store_true', help='Use two layers of FC layer for channel decision')
parser.add_argument('--input_mask', action='store_true', help='Add input mask for BasicBlock')

# model
parser.add_argument('--model', type=str, default='resnet101', help='network model name')
parser.add_argument('--model_cfg', type=str, default='baseline', help='network configuration')
parser.add_argument('--resolution_mask', action='store_true', help='share a mask within a same resolution')
parser.add_argument('--mask_type', type=str, default='conv', help='Type of mask')

# gumbel args
parser.add_argument('--remove_gumbel', default=0.8, type=float, help='The time stage for removing gumbel noise')
parser.add_argument('--gumbel_interval', nargs="+", type=float, default=[0.5, 0.8, 1], help='gumbel interval')
parser.add_argument('--gumbel_temp', nargs="+", type=float, default=[5, 2.5, 1], help='gumbel Temperature value')

# file management
parser.add_argument('-s', '--save_dir', type=str, default='', help='directory to save model')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--load', type=str, default='', help='load model path')
parser.add_argument('--auto_resume', action='store_true', help='plot ponder cost')

# evaluation
parser.add_argument('-e', '--evaluate', action='store_true', help='evaluation mode')


args_cifar10 = parser.parse_args()
