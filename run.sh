#!/bin/sh
NAME=DilatedCNN_kernel3_deepskip_feat_large_subavg
mkdir -p outputs/$NAME
nohup python -u train.py \
--save_name $NAME \
--log_dir $NAME \
--device_id 1 \
> outputs/$NAME/output.log 2>&1 &
