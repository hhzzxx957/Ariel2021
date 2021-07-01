#!/bin/sh
NAME=resnet_featquan_preprocessed_adamw_cross
mkdir -p outputs/$NAME
nohup python -u train.py \
--save_name $NAME \
--log_dir $NAME \
--device_id 3 \
> outputs/$NAME/output.log 2>&1 &
