#!/bin/bash
cd /media/hdd1t/baoheng/DPCAS_ARTIFACT/software
source activate
conda activate aec
sh scripts/evaluate_imagenet.sh /media/ssd0/imagenet/ 48
