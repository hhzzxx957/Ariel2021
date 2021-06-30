Name=featquan_proprocessed6_adamw_cross
nohup python -u prediction.py --save_name $Name --device_id 0 >> outputs/$Name/output.log 2>&1 &
