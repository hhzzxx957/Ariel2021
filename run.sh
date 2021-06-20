#!/bin/sh
NAME=DilatedCNN_kernel3_avgpool_lr05_cycle_deep_feat_small_mem
mkdir -p outputs/$NAME
nohup python -u train.py \
--save_name $NAME \
--log_dir $NAME \
--device_id 1 \
> outputs/$NAME/output.log 2>&1 &
