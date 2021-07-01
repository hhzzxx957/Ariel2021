Name=featquanphoton_proprocessed10_adamw_cross_seed1024
nohup python -u prediction.py --save_name $Name --device_id 0 >> outputs/$Name/output.log 2>&1 &
