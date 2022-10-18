# Hardware Source

The subfolder contains the hls source code and vivado project tcl files for resnet basicblock (18, 34) and bottleneck (50, 101) with both dense baseline and dynamic pruning version DPUnit. 

To reproduce the hardware design, follow the instructions in the Readme.md in each subfolder.

## Expect Performance on ZCU 102

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
| ResNet-18 | Parallel | 14.3 ms |
| ResNet-34 | Serial | 48.9 ms |
| ResNet-34 | Parallel | 24.5 ms |


### Basicblock DPUnit
| Model | Implementation | s25-c25 | s50-c50 | s75-c75| 
| --- | --- | --- | --- | --- |
| ResNet-18 | Serial | 22.1 ms | 17.1ms | 13.3 ms |
| ResNet-18 | Parallel | 16.3 ms | 14.7ms | 13.3 ms |
| ResNet-34 | Serial | 39.4 ms | 28.3ms | 20.1 ms |
| ResNet-34 | Parallel | 27.2 ms | 23.3ms | 20.1 ms |



