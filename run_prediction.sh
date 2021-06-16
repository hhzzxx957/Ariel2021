Name=DilatedCNN_kernel3
nohup bash -c "python -u prediction.py --save_name $Name" >> outputs/$Name/output.log 2>&1 &