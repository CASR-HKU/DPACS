# -*-coding:utf-8-*-

cmds = [

    # RESNET101 BLOCK-MASK
    "python main_imagenet.py --dataset-root /media/ssd0/imagenet --pooling_method ave --budget 0.50 --group_size 64 --channel_budget 0.50 --channel_stage 2 3 --batchsize 48 --model_cfg hardware_2048 --load DPACS_checkpoint/imagenet_resnet101_BlockMask/s50_c50.pth -e",
    "python main_imagenet.py --dataset-root /media/ssd0/imagenet --pooling_method ave --budget 0.75 --group_size 64 --channel_budget 0.75 --channel_stage 2 3 --batchsize 48 --model_cfg hardware_2048 --load DPACS_checkpoint/imagenet_resnet101_BlockMask/s75_c75.pth -e",

    # RESNET101 RESOLUTION-MASK
    "python main_imagenet.py --dataset-root /media/ssd0/imagenet --pooling_method ave --budget 0.50 --group_size 64 --channel_budget 0.50 --channel_stage 2 3 --batchsize 48 --model_cfg hardware_2048 --resolution_mask --load DPACS_checkpoint/imagenet_resnet101_ResolutionMask/s50_c50.pth -e",
    "python main_imagenet.py --dataset-root /media/ssd0/imagenet --pooling_method ave --budget 0.75 --group_size 64 --channel_budget 0.75 --channel_stage 2 3 --batchsize 48 --model_cfg hardware_2048 --resolution_mask --load DPACS_checkpoint/imagenet_resnet101_ResolutionMask/s75_c75.pth -e",

    # RESNET50 BLOCK-MASK
    "python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize 72 --pooling_method ave --budget 0.75  --group_size 64 --channel_budget 0.75 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s75_c75.pth -e --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize 72 --pooling_method ave --budget 0.75  --group_size 64 --channel_budget 0.50 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s75_c50.pth -e --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize 72 --pooling_method ave --budget 0.75  --group_size 64 --channel_budget 0.25 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s75_c25.pth -e --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize 72 --pooling_method ave --budget 0.50  --group_size 64 --channel_budget 0.75 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s50_c75.pth -e --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize 72 --pooling_method ave --budget 0.50  --group_size 64 --channel_budget 0.50 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s50_c50.pth -e --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize 72 --pooling_method ave --budget 0.50  --group_size 64 --channel_budget 0.25 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s50_c25.pth -e --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize 72 --pooling_method ave --budget 0.25  --group_size 64 --channel_budget 0.75 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s25_c75.pth -e --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize 72 --pooling_method ave --budget 0.25  --group_size 64 --channel_budget 0.50 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s25_c50.pth -e --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize 72 --pooling_method ave --budget 0.25  --group_size 64 --channel_budget 0.25 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s25_c25.pth -e --dataset-root /media/ssd0/imagenet",

    # RESNET50 RESOLUTION-MASK
    "python main_imagenet.py --model resnet50 --resolution_mask --model_cfg hardware_2048 --batchsize 72 --pooling_method ave --budget 0.75  --group_size 64 --channel_budget 0.75 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_ResolutionMask/s75_c75_resmask.pth -e --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py --model resnet50 --resolution_mask --model_cfg hardware_2048 --batchsize 72 --pooling_method ave --budget 0.50  --group_size 64 --channel_budget 0.50 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_ResolutionMask/s50_c50_resmask.pth -e --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py --model resnet50 --resolution_mask --model_cfg hardware_2048 --batchsize 72 --pooling_method ave --budget 0.25  --group_size 64 --channel_budget 0.25 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_ResolutionMask/s25_c25_resmask.pth -e --dataset-root /media/ssd0/imagenet",

    # RESNET18 INPUT MASK
    "python main_imagenet.py --model resnet18 --pooling_method ave --budget 0.5 --group_size 64 --channel_budget 0.5 --channel_stage 2 3 --batchsize 256 -e --load DPACS_checkpoint/imagenet_basicBlock_inputMask/res18_s50_c50.pth --input_mask --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py --model resnet18 --pooling_method ave --budget 0.75 --group_size 64 --channel_budget 0.5 --channel_stage 2 3 --batchsize 256 -e --load DPACS_checkpoint/imagenet_basicBlock_inputMask/res18_s75_c50.pth --input_mask --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py --model resnet18 --pooling_method ave --budget 0.75 --group_size 64 --channel_budget 0.75 --channel_stage 2 3 --batchsize 256 -e --load DPACS_checkpoint/imagenet_basicBlock_inputMask/res18_s75_c75.pth --input_mask --dataset-root /media/ssd0/imagenet",
    
    # RESNET34 INPUT MASK
    "python main_imagenet.py --model resnet34 --pooling_method ave --budget 0.50 --group_size 64 --channel_budget 0.5 --channel_stage 2 3 --batchsize 256 -e --load DPACS_checkpoint/imagenet_basicBlock_inputMask/res34_s50_c50.pth --input_mask --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py --model resnet34 --pooling_method ave --budget 0.75 --group_size 64 --channel_budget 0.5 --channel_stage 2 3 --batchsize 256 -e --load DPACS_checkpoint/imagenet_basicBlock_inputMask/res34_s75_c50.pth --input_mask --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py --model resnet34 --pooling_method ave --budget 0.75 --group_size 64 --channel_budget 0.75 --channel_stage 2 3 --batchsize 256 -e --load DPACS_checkpoint/imagenet_basicBlock_inputMask/res34_s75_c75.pth --input_mask --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py --model resnet34 --pooling_method ave --budget 0.75 --group_size 64 --channel_budget 0.25 --channel_stage 2 3 --batchsize 256 -e --load DPACS_checkpoint/imagenet_basicBlock_inputMask/res34_s75_c25.pth --input_mask --dataset-root /media/ssd0/imagenet",

    # Baseline
    "python main_imagenet.py -e --model resnet18 --budget -1 --load DPACS_checkpoint/baseline/resnet18-5c106cde.pth --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py -e --model resnet34 --budget -1 --load DPACS_checkpoint/baseline/resnet34-333f7ec4.pth --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py -e --model resnet50 --budget -1 --load DPACS_checkpoint/baseline/resnet50-19c8e357.pth --dataset-root /media/ssd0/imagenet",
    "python main_imagenet.py -e --model resnet101 --budget -1 --load DPACS_checkpoint/baseline/resnet101-5d3b4d8f.pth --dataset-root /media/ssd0/imagenet",

    # Mobilenet
    "python main_imagenet.py -e --model MobileNetV2 --batchsize 64 --pooling_method ave --budget 0.75 --group_size 32 --channel_budget 0.75 --channel_stage 6 -1 --load DPACS_checkpoint/imagenet_mobilenet/s75_c75.pth --dataset-root /media/ssd0/imagenet",

    # Cifar10
    "python main_cifar.py -e --model resnet32_BN --pooling_method ave --budget 0.75 --load DPACS_checkpoint/cifar10_resnet32_BN/s75_c50.pth --group_size 8 --batchsize 320 --channel_budget 0.50 --channel_stage 1 2",
    "python main_cifar.py -e --model resnet32_BN --pooling_method ave --budget -1 --load DPACS_checkpoint/baseline/cifar10_resnet32_BN.pth",
]

import os

for idx, cmd in enumerate(cmds):
    os.system(cmd)
