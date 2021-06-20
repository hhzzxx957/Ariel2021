Name=DilatedCNN_kernel3_avgpool_lr05_cycle_deep_feat_small_mem
nohup python -u prediction.py --save_name $Name --device_id 0 >> outputs/$Name/output.log 2>&1 &
