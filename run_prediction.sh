Name=DilatedCNN_kernel3_batch512_l1
nohup python -u prediction.py --save_name $Name --device_id 0 >> outputs/$Name/output.log 2>&1 &