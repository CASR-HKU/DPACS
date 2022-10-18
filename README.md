# DPACS: Hardware Accelerated Dynamic Neural Network Pruning through Algorithm-Architecture Co-design

This repository contains the implementation for 

> **DPACS: Hardware Accelerated Dynamic Neural Network Pruning through Algorithm-Architecture Co-design**  
> Yizhao Gao, Baoheng Zhang, Xiaojuan Qi, Hayden So  
> (ASPLOP 2023) 

DPCAS is an algorithm-architecture co-design framework for dynamic neural network pruning. It utilizes a hardware-aware dynamic spatial and channel pruning mechanism in conjunction with a dynamic dataflow engine in hardware to facilitate efficient processing of the pruned network. 

## Outlines
* `./hardware`: Hardware source code, bitstreams, driver of the DPACS accelerator
* `./software`: Python source code, scripts, model checkpoints and training logs DPACS alrgoithm

## Dependencies
The dependancies of the hardware and software experiments are specified in `./hardware/README.md` and `./software/README.md` respectively.


