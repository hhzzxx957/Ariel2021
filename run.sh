#!/bin/sh
NAME=DilatedCNN_kernel3_batch512_avgpool_sepconv
mkdir -p outputs/$NAME
nohup python -u train.py \
--save_name $NAME \
--log_dir $NAME \
--device_id 3 \
> outputs/$NAME/output.log 2>&1 &
# python -u prediction.py --save_name $NAME