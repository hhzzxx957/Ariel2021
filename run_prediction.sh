Name=DilatedCNN_kernel3_avgpool_lr05_cycle_deep_feat_large_subavg
nohup python -u prediction.py --save_name $Name --device_id 0 >> outputs/$Name/output.log 2>&1 &
