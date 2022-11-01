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
| network        | impl     | baseline | s25-c25  | s50-c25  | s75-c25  | s25-c50  | s50-c50  | s75-c50  | s25-c75  | s50-c75  | s75-c75  |
| ---------- | -------- | -------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| resnet 18  | serial   | 25.487   | 21.892 | 17.557 | 14.37  | 21.028 | 17.513 | 13.979 | 20.731 | 16.615 | 13.244 |
| resnet 18  | parallel | 14.202   | 14.309 | 13.647 | 12.865 | 13.898 | 13.238 | 12.342 | 13.43  | 12.637 | 12.049 |
| resnet 34  | serial   | 48.681   | 39.286 | 29.744 | 22.705 | 37.242 | 28.061 | 21.338 | 35.519 | 26.602 | 19.99  |
| resnet 34  | parallel | 24.316   | 23.708 | 21.822 | 20.154 | 22.178 | 20.664 | 18.815 | 20.834 | 19.225 | 17.836 |
| resnet 50  | serial   | 58.811   | 44.328 | 33.676 | 26.354 | 41.875 | 31.972 | 25.189 | 40.264 | 30.515 | 24.358 |
| resnet 50  | parallel | 32.839   | 30.913 | 27.401 | 24.797 | 29.628 | 27.297 | 25.239 | 28.929 | 26.444 | 24.675 |
| resnet 101 | serial   | 95.325   | 62.112 | 45.766 | 35.983 | 53.834 | 40.712 | 32.181 | 46.74  | 36.098 | 29.045 |
| resnet 101 | parallel | 49.475   | 43.418 | 39.453 | 35.933 | 38.509 | 35.279 | 32.061 | 35.128 | 32.099 | 29.481 |



### Bottleneck DPUnit with Resolution Mask (ms)
| Model | Implementation | s25-c25 | s50-c50 | s75-c75| 
| --- | --- | --- | --- | --- |
| ResNet-50 | Serial | 44.7 | 32.1 | 23.1 |
| ResNet-50 | Parallel | 30.9 | 26.1 | 22.5 |
| ResNet-101 | Serial | 65.4 | 41.1 | 27.1 |
| ResNet-101 | Parallel | 43.6 | 33.8 | 26.1 |




