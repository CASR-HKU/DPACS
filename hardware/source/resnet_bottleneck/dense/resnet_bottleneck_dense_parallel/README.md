# ResNet Bottleneck Dense Accelerator with Parallel PE Implementation

The dataflow accelerator is mainly composed of 3 convolution kernel:
- Conv 1x1 with input channel: IC_0, output channel: OC_0, skip signal skip_0
- Conv 3x3 with input channel: OC_0, output channel: OC_3, skip signal skip_3
- Conv 1x1 with input channel: OC_3, output channel: OC_1, skip signal skip_1

If the skip signal is marked as 1, the correpsonding computation of the kernel will be bypassed. 
Other control signals are described in top.cpp . 

## Makefile targets
- tb_gen: Generate testbench data using PyTorch
- csim: Run C-simulation using the testbench data
- hls: Synthesis the RTL design using HLS, C-RTL co-simulation
- bitsteam: Synthesis, Implementation, Write Bitstream of the design
- unpack: unzip output bitstream and hardware description
- all: Run all the way to unpack

## Run hardware synthesis
Note that you need to add Vivado tools to your environment before running the makefile by
```
    source /path/to/xilinx/2020.2/settings64.sh
```

To regenerate hardware design, run the following command:
```
    make all
```
The output bitstream will also be copied to the drive folder. Log will be recorded in the ./log folder.
