#!/bin/sh
NAME=DilatedCNN_kernel3_deep_feat_large_subavg_metricadamw
mkdir -p outputs/$NAME
nohup python -u train.py \
--save_name $NAME \
--log_dir $NAME \
--device_id 3 \
> outputs/$NAME/output.log 2>&1 &
