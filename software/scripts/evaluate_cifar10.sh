python main_cifar.py -e --model resnet32_BN --pooling_method ave --budget -1 --load DPACS_checkpoint/baseline/cifar10_resnet32_BN.pth --batchsize $1
python main_cifar.py -e --model resnet32_BN --pooling_method ave --budget 0.75 --load DPACS_checkpoint/cifar10_resnet32_BN/s75_c50.pth --group_size 8 --batchsize $1 --channel_budget 0.50 --channel_stage 1 2
