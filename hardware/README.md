# DPCAS Hardware

This folder contains the hardware implementation of DPCAS accelerator called DPUnit. It's a stream-based dataflow architecture supporting dynamic pruning of a residual block. It's developed using Vitis HLS and Vivado Design Suite. The design is implemented on a ZCU 102 development board with a Zynq UltraScale+ MPSoC XCZU3EG device.    

## Dependancy 

- Ubuntu 18.04
- [Vivado 2020.2](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools/archive.html) 
    - Install Vivado using the official instructions. 
    - source /path/to/xilinx/Vivado/2020.2/settings64.sh after installation. 
- python 3 with numpy and PyTorch

## Hardware Settings

The ZCU development board is powered by [PYNQ 2.6](https://pynq.readthedocs.io/en/v2.6.1/index.html).
Since there is no official pre-built image for ZCU 102, you might need to build a image by yourself. You can refer to the instructions in these websites:

- [pynq.io](http://www.pynq.io/board.html)
- [PYNQ SD Card Image](https://pynq.readthedocs.io/en/latest/pynq_sd_card.html)

## Outlines

Directory | Description
----------| -----------
source/resnet_basicblock | Source code for the hardware design of resnet 18/34
source/resnet_bottleneck | Source code for the hardware design of resnet 50/101
drive | Contains bitstreams and jupyter notebooks to run on-board hardware test

## Expected Performance on ZCU 102
### Bottleneck Dense Baseline
| Model | Implementation | Latency | 
| --- | --- | --- |
| ResNet-50 | Serial | 59.0 ms |
| ResNet-50 | Parallel | 33.1 ms |
| ResNet-101 | Serial | 95.7 ms |
| ResNet-101 | Parallel | 49.9 ms |


### Bottleneck DPUnit
| Model | Implementation | s25-c25 | s50-c50 | s75-c75| 
| --- | --- | --- | --- | --- |
| ResNet-50 | Serial | 44.5 ms | 31.9 ms | 24.5 ms |
| ResNet-50 | Parallel | 31.3 ms | 27.5 ms | 24.7 ms |
| ResNet-101 | Serial | 62.7 ms | 40.1 ms | 29.3 ms |
| ResNet-101 | Parallel | 43.8 ms | 34.9 ms | 29.8 ms |


### Bottleneck DPUnit with Resolution Mask
| Model | Implementation | s25-c25 | s50-c50 | s75-c75| 
| --- | --- | --- | --- | --- |
| ResNet-50 | Serial | 44.7 ms | 32.1 ms | 23.1 ms |
| ResNet-50 | Parallel | 30.9 ms | 26.1 ms | 22.5 ms |
| ResNet-101 | Serial | 65.4 ms | 41.1 ms | 27.1 ms |
| ResNet-101 | Parallel | 43.6 ms | 33.8 ms | 26.1 ms |


### Basicblock Dense Baseline
| Model | Implementation | Latency | 
| --- | --- | --- |
| ResNet-18 | Serial | 25.6 ms |
| ResNet-34 | Serial | 48.9 ms |


### Basicblock DPUnit
| Model | Implementation | s25-c25 | s50-c50 | s75-c75| 
| --- | --- | --- | --- | --- |
| ResNet-18 | Serial | 22.1 ms | 17.1ms | 13.3 ms |
| ResNet-34 | Serial | 39.4 ms | 28.3ms | 20.1 ms |
