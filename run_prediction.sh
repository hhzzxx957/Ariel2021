Name=featquan_proprocessed_fft_adamw
nohup python -u prediction.py --save_name $Name --device_id 0 >> outputs/$Name/output.log 2>&1 &
