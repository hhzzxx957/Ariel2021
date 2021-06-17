# %%
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import ArielMLDataset
from constants import *

data_dir = "/data-nbd/ml_dataset/timeseries/pkdd_ml_data_challenge2"
#%%
def get_train_features():
    lc_train_path = data_dir+ "/data/noisy_train/home/ucapats/Scratch/ml_data_challenge/training_set/noisy_train"
    files = sorted([p for p in os.listdir(lc_train_path) if p.endswith('txt')])

    output = []
    for file in tqdm(files):
        with open(os.path.join(lc_train_path,file), 'r') as f:
            properties = f.readlines()[:6]
            res = []
            for s in properties:
                res.append(float(s.split(': ')[1]))
            output.append(res)

    df = pd.DataFrame(output, columns=['star_temp', 'star_logg', 'star_rad', 'star_mass', 'star_k_mag', 'period'])
    df.to_csv('data/properties_df.csv')


# %%
def get_test_features():
    lc_test_path = data_dir+"/data/noisy_test/home/ucapats/Scratch/ml_data_challenge/test_set/noisy_test"
    files = sorted([p for p in os.listdir(lc_test_path) if p.endswith('txt')])

    output = []
    for file in tqdm(files):
        with open(os.path.join(lc_test_path, file), 'r') as f:
            properties = f.readlines()[:6]
            res = []
            for s in properties:
                res.append(float(s.split(': ')[1]))
            output.append(res)

    df = pd.DataFrame(output, columns=['star_temp', 'star_logg', 'star_rad', 'star_mass', 'star_k_mag', 'period'])
    df.to_csv('data/test_properties_df.csv')
get_test_features()
# %%
files = sorted([p for p in os.listdir(lc_train_path) if p.endswith('txt')])

dataset = ArielMLDataset(lc_train_path, params_train_path, shuffle=False)
output = np.zeros(dataset[0]['lc'].T.detach().numpy().shape)
tot_len = len(dataset)
for i in tqdm(range(tot_len)):
    output += dataset[i]['lc'].T.detach().numpy()
output /= tot_len
res = np.mean(output, axis=1)
print(res.shape)
np.save('data/average_signal.npy')
# %%
