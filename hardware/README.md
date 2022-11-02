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
| net       | impl     | baseline | 25-25  | 50-25  | 75-25  | 25-50  | 50-50  | 75-50  | 25-75  | 50-75  | 75-75  |
| --------- | -------- | -------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| resnet 18 | serial   | 29.872   | 24.266 | 19.24  | 14.653 | 23.566 | 18.121 | 14.046 | 22.841 | 17.88  | 13.541 |
| resnet 18 | parallel | 15.748   | 15.636 | 14.874 | 13.934 | 15.188 | 14.476 | 13.547 | 14.782 | 13.957 | 13.18  |
| resnet 34 | serial   | 53.046   | 40.567 | 29.624 | 22.314 | 38.375 | 28.556 | 20.654 | 36.532 | 26.456 | 19.236 |
| resnet 34 | parallel | 26.029   | 24.93  | 23.095 | 21.268 | 23.552 | 21.684 | 20.106 | 22.099 | 20.465 | 18.962 |
| resnet 50 | serial   | 58.811   | 44.328 | 33.676 | 26.354 | 41.875 | 31.972 | 25.189 | 40.264 | 30.515 | 24.358 |
| resnet 50 | parallel | 32.839   | 30.913 | 27.401 | 24.797 | 29.628 | 27.297 | 25.239 | 28.929 | 26.444 | 24.675 |
| resnet101 | serial   | 95.325   | 62.112 | 45.766 | 35.983 | 53.834 | 40.712 | 32.181 | 46.74  | 36.098 | 29.045 |
| resnet101 | parallel | 49.475   | 43.418 | 39.453 | 35.933 | 38.509 | 35.279 | 32.061 | 35.128 | 32.099 | 29.481 |



### Bottleneck DPUnit with Resolution Mask (ms)
| Model | Implementation | s25-c25 | s50-c50 | s75-c75| 
| --- | --- | --- | --- | --- |
| ResNet-50 | Serial | 44.7 | 32.1 | 23.1 |
| ResNet-50 | Parallel | 30.9 | 26.1 | 22.5 |
| ResNet-101 | Serial | 65.4 | 41.1 | 27.1 |
| ResNet-101 | Parallel | 43.6 | 33.8 | 26.1 |




