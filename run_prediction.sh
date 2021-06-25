Name=DilatedCNN_kernel3_deep_featlgb_large_preprocessed_metricadamw
nohup python -u prediction.py --save_name $Name --device_id 0 >> outputs/$Name/output.log 2>&1 &
