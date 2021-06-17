#!/bin/sh
NAME=DilatedCNN_kernel3_avgpool_lr1_cycle
mkdir -p outputs/$NAME
nohup python -u train.py \
--save_name $NAME \
--log_dir $NAME \
--device_id 2 \
> outputs/$NAME/output.log 2>&1 &
