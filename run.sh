#!/bin/sh
NAME=featquan_preprocessed_adamw_seed100
mkdir -p outputs/$NAME
nohup python -u train.py \
--save_name $NAME \
--log_dir $NAME \
--device_id 1 \
> outputs/$NAME/output.log 2>&1 &
