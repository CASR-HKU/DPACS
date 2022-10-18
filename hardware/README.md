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
resnet_basicblock | Source code for the hardware design of resnet 18/34
resnet_bottleneck | Source code for the hardware design of resnet 50/101
drive | Contains bitstreams and jupyter notebooks to run on-board hardware test

