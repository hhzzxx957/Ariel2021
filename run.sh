mkdir -p outputs/MLP_10000
nohup bash -c "python -u train.py --save_name MLP_10000" > outputs/MLP_10000/output.log 2>&1 &
# python -u prediction.py --save_name MLP_10000