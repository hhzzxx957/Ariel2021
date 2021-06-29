#!/bin/sh
NAME=featquan_proprocessed_variancefeatnorm_adamw
mkdir -p outputs/$NAME
nohup python -u train.py \
--save_name $NAME \
--log_dir $NAME \
--device_id 3 \
> outputs/$NAME/output.log 2>&1 &
