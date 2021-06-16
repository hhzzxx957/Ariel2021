#!/bin/sh
NAME=DilatedCNN_kernel3
mkdir -p outputs/$NAME
nohup bash -c \
"python -u train.py --save_name $NAME --log_dir $NAME" \
> outputs/$NAME/output.log 2>&1 &
# python -u prediction.py --save_name $NAME