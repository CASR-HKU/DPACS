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

## Expected Lantency Performance on ZCU 102 (ms)
| net       | impl     | baseline | 25-25  | 50-25 | 75-25 | 25-50 | 50-50 | 75-50 | 25-75 | 50-75 | 75-75 |
| --------- | -------- | -------- | ------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| resnet 18 | serial   | 29.87    | 23.35  | 18.28 | 14.05 | 21.41 | 16.89 | 12.47 | 19.71 | 15.04 | 11.22 |
| resnet 18 | parallel | 15.75    | 14.82  | 13.95 | 13.07 | 13.27 | 12.53 | 12.02 | 11.78 | 11.37 | 10.80 |
| resnet 34 | serial   | 53.05    | 38.19  | 28.70 | 21.04 | 34.53 | 25.38 | 18.21 | 31.18 | 22.22 | 15.98 |
| resnet 34 | parallel | 26.03    | 23.28  | 21.63 | 19.97 | 20.54 | 19.00 | 17.70 | 17.51 | 16.36 | 15.08 |
| resnet 50 | serial   | 58.81    | 41.51  | 31.12 | 24.47 | 36.45 | 27.54 | 21.73 | 32.94 | 24.49 | 19.62 |
| resnet 50 | parallel | 32.84    | 28.48  | 26.38 | 24.30 | 24.97 | 23.24 | 21.86 | 22.12 | 20.88 | 19.90 |
| resnet101 | serial   | 95.33    | 59.60  | 44.35 | 34.15 | 48.58 | 36.21 | 28.48 | 39.38 | 30.27 | 24.45 |
| resnet101 | parallel | 49.48    | 40.88  | 37.03 | 33.99 | 33.56 | 30.87 | 28.38 | 28.38 | 26.32 | 24.59 |



### Bottleneck DPUnit with Resolution Mask (ms)
| Model | Implementation | s25-c25 | s50-c50 | s75-c75| 
| --- | --- | --- | --- | --- |
| ResNet-50 | Serial | 41.6 | 27.5 | 18.4 |
| ResNet-50 | Parallel | 28.0 | 21.7 | 17.3 |
| ResNet-101 | Serial | 60.6 | 36.2 | 22.4 |
| ResNet-101 | Parallel | 40.9 | 29.4 | 21.3 |




