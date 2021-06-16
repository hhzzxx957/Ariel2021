# %%
import pandas as pd
import numpy as np
res = np.loadtxt('/data-nbd/ml_dataset/timeseries/pkdd_ml_data_challenge2/outputs/baseline/evaluation_2021-06-15.txt')
print(res.shape)
# %%
from models import MLP
model = MLP(num_mlp_layers = 3, emb_dim = 400)
