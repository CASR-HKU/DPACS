python main_imagenet.py --dataset-root $1 --pooling_method ave --budget 0.50 --group_size 64 --channel_budget 0.50 --channel_stage 2 3 --batchsize $2 --model_cfg hardware_2048 --load DPACS_checkpoint/imagenet_resnet101_BlockMask/s50_c50.pth -e
python main_imagenet.py --dataset-root $1 --pooling_method ave --budget 0.75 --group_size 64 --channel_budget 0.75 --channel_stage 2 3 --batchsize $2 --model_cfg hardware_2048 --load DPACS_checkpoint/imagenet_resnet101_BlockMask/s75_c75.pth -e
$2
python main_imagenet.py --dataset-root $1 --pooling_method ave --budget 0.50 --group_size 64 --channel_budget 0.50 --channel_stage 2 3 --batchsize $2 --model_cfg hardware_2048 --resolution_mask --load DPACS_checkpoint/imagenet_resnet101_ResolutionMask/s50_c50.pth -e
python main_imagenet.py --dataset-root $1 --pooling_method ave --budget 0.75 --group_size 64 --channel_budget 0.75 --channel_stage 2 3 --batchsize $2 --model_cfg hardware_2048 --resolution_mask --load DPACS_checkpoint/imagenet_resnet101_ResolutionMask/s75_c75.pth -e

python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize $2 --pooling_method ave --budget 0.75  --group_size 64 --batchsize $2 --channel_budget 0.75 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s75_c75.pth -e --dataset-root $1
python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize $2 --pooling_method ave --budget 0.75  --group_size 64 --batchsize $2 --channel_budget 0.50 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s75_c50.pth -e --dataset-root $1
python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize $2 --pooling_method ave --budget 0.75  --group_size 64 --batchsize $2 --channel_budget 0.25 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s75_c25.pth -e --dataset-root $1
python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize $2 --pooling_method ave --budget 0.50  --group_size 64 --batchsize $2 --channel_budget 0.75 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s50_c75.pth -e --dataset-root $1
python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize $2 --pooling_method ave --budget 0.50  --group_size 64 --batchsize $2 --channel_budget 0.50 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s50_c50.pth -e --dataset-root $1
python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize $2 --pooling_method ave --budget 0.50  --group_size 64 --batchsize $2 --channel_budget 0.25 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s50_c25.pth -e --dataset-root $1
python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize $2 --pooling_method ave --budget 0.25  --group_size 64 --batchsize $2 --channel_budget 0.75 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s25_c75.pth -e --dataset-root $1
python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize $2 --pooling_method ave --budget 0.25  --group_size 64 --batchsize $2 --channel_budget 0.50 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s25_c50.pth -e --dataset-root $1
python main_imagenet.py --model resnet50 --model_cfg hardware_2048 --batchsize $2 --pooling_method ave --budget 0.25  --group_size 64 --batchsize $2 --channel_budget 0.25 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_BlockMask/s25_c25.pth -e --dataset-root $1

python main_imagenet.py --model resnet50 --resolution_mask --model_cfg hardware_2048 --batchsize $2 --pooling_method ave --budget 0.75 --batchsize $2 --group_size 64 --channel_budget 0.75 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_ResolutionMask/s75_c75.pth -e --dataset-root $1
python main_imagenet.py --model resnet50 --resolution_mask --model_cfg hardware_2048 --batchsize $2 --pooling_method ave --budget 0.50 --batchsize $2 --group_size 64 --channel_budget 0.50 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_ResolutionMask/s50_c50.pth -e --dataset-root $1
python main_imagenet.py --model resnet50 --resolution_mask --model_cfg hardware_2048 --batchsize $2 --pooling_method ave --budget 0.25 --batchsize $2 --group_size 64 --channel_budget 0.25 --channel_stage 2 3 --load DPACS_checkpoint/imagenet_resnet50_ResolutionMask/s25_c25.pth -e --dataset-root $1

python main_imagenet.py --model resnet18 --pooling_method ave --batchsize $2 --budget 0.5 --group_size 64 --channel_budget 0.5 --channel_stage 2 3 --batchsize $2 -e --load DPACS_checkpoint/imagenet_basicBlock_inputMask/res18_s50_c50.pth --input_mask --dataset-root $1
python main_imagenet.py --model resnet18 --pooling_method ave --batchsize $2 --budget 0.75 --group_size 64 --channel_budget 0.5 --channel_stage 2 3 --batchsize $2 -e --load DPACS_checkpoint/imagenet_basicBlock_inputMask/res18_s75_c50.pth --input_mask --dataset-root $1
python main_imagenet.py --model resnet18 --pooling_method ave --batchsize $2 --budget 0.75 --group_size 64 --channel_budget 0.75 --channel_stage 2 3 --batchsize $2 -e --load DPACS_checkpoint/imagenet_basicBlock_inputMask/res18_s75_c75.pth --input_mask --dataset-root $1

python main_imagenet.py --model resnet34 --pooling_method ave --batchsize $2 --budget 0.50 --group_size 64 --channel_budget 0.5 --channel_stage 2 3 --batchsize $2 -e --load DPACS_checkpoint/imagenet_basicBlock_inputMask/res34_s50_c50.pth --input_mask --dataset-root $1
python main_imagenet.py --model resnet34 --pooling_method ave --batchsize $2 --budget 0.75 --group_size 64 --channel_budget 0.5 --channel_stage 2 3 --batchsize $2 -e --load DPACS_checkpoint/imagenet_basicBlock_inputMask/res34_s75_c50.pth --input_mask --dataset-root $1
python main_imagenet.py --model resnet34 --pooling_method ave --batchsize $2 --budget 0.75 --group_size 64 --channel_budget 0.75 --channel_stage 2 3 --batchsize $2 -e --load DPACS_checkpoint/imagenet_basicBlock_inputMask/res34_s75_c75.pth --input_mask --dataset-root $1
python main_imagenet.py --model resnet34 --pooling_method ave --batchsize $2 --budget 0.75 --group_size 64 --channel_budget 0.25 --channel_stage 2 3 --batchsize $2 -e --load DPACS_checkpoint/imagenet_basicBlock_inputMask/res34_s75_c25.pth --input_mask --dataset-root $1

python main_imagenet.py -e --model MobileNetV2 --batchsize $2 --pooling_method ave --budget 0.75 --group_size 32 --channel_budget 0.75 --channel_stage 6 -1 --load DPACS_checkpoint/imagenet_mobilenet/s75_c75.pth --dataset-root $1

python main_imagenet.py -e --batchsize $2 --model resnet18 --budget -1 --load DPACS_checkpoint/baseline/resnet18-5c106cde.pth --dataset-root $1
python main_imagenet.py -e --batchsize $2 --model resnet34 --budget -1 --load DPACS_checkpoint/baseline/resnet34-333f7ec4.pth --dataset-root $1
python main_imagenet.py -e --batchsize $2 --model resnet50 --budget -1 --load DPACS_checkpoint/baseline/resnet50-19c8e357.pth --dataset-root $1
python main_imagenet.py -e --batchsize $2 --model resnet101 --budget -1 --load DPACS_checkpoint/baseline/resnet101-5d3b4d8f.pth --dataset-root $1
