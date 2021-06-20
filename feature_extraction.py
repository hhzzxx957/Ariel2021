# %%
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import ArielMLDataset
from joblib import Parallel, delayed
import torch
import gc
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
np.save('data/average_signal.npy', res)
# %%
dataset = ArielMLDataset(lc_train_path, params_train_path, shuffle=False)
output = []

tot_len = len(dataset)
for i in tqdm(range(tot_len)): #tot_len
    file_data = np.concatenate([
        dataset[i]['lc'].detach().numpy(),
        dataset[i]['target'].reshape(-1, 1).detach().numpy()
    ],
                               axis=1)
    output.append(file_data)
    gc.collect()
res = np.concatenate(output, axis=0)

print(res.shape)
np.save('data/full_train_signal.npy', res)

#%%
dataset = ArielMLDataset(lc_train_path, params_train_path, shuffle=False)
tot_len = len(dataset)
def extract_signal(i):
    file_data = np.concatenate([
        dataset[i]['lc'].detach().numpy(),
        dataset[i]['target'].reshape(-1, 1).detach().numpy()
    ],
                            axis=1)
    return file_data

output = Parallel(n_jobs=20)(delayed(extract_signal)(i) for i in tqdm(range(tot_len)))
#%%
dataset = ArielMLDataset(lc_test_path, shuffle=False)
tot_len = len(dataset)
def extract_signal(i):
    return dataset[i]['lc'].detach().numpy()

output = Parallel(n_jobs=20)(delayed(extract_signal)(i) for i in tqdm(range(tot_len)))
#%%
res = np.concatenate(output, axis=0)

print(res.shape)
np.save('data/full_test_signal.npy', res)

# %%
ids = []
for i in range(125600, 125600+53900):
    ids.extend([i]*55)

positions = list(range(55))*(tot_len)
print(len(ids), len(positions))

df = pd.DataFrame(res)
df['id'] = ids
df['position'] = positions
# %%
colnames = list(df.columns)
# colnames[-3] = 'label'
df.columns = colnames

# %%
# feat_df = pd.read_csv('data/properties_df.csv')
feat_df = pd.read_csv('data/test_properties_df.csv')
# %%
feat_df['id'] = list(range(125600, 125600+53900)) # 125600
feat_df.drop('Unnamed: 0', axis=1, inplace=True)
df_all = pd.merge(df, feat_df, on='id')
# %%
df_all.to_pickle('data/full_test_df.pickle')

# %%
# get average std
signal = torch.load('data/full_train_signal.pt')
avg = np.load('data/average_signal.npy')

sub_signal = signal[:,:300]-avg
std_signal = np.std(sub_signal.numpy(), axis=1)
np.mean(std_signal)
# %%
pred1 = np.loadtxt('outputs/DilatedCNN_kernel3_avgpool_lr05_cycle_deep_feat_large/evaluation_2021-06-19_backup.txt')
pred2 = np.loadtxt('outputs/DilatedCNN_kernel3_avgpool_lr05_cycle_deep_feat_large/evaluation_2021-06-19.txt')
pred3 = np.loadtxt('outputs/DilatedCNN_kernel3_avgpool_lr05_cycle_skip_large/evaluation_2021-06-18.txt')
# %%
pred4 = np.loadtxt('outputs/DilatedCNN_kernel3_avgpool_lr05_cycle_deep_feat_small_subavg/evaluation_2021-06-19.txt') 
pred5 = np.loadtxt('outputs/DilatedCNN_kernel3_avgpool_lr05_cycle_deep_feat_large_subavg/evaluation_2021-06-19.txt') 

# %%
test_signal = torch.load('data/full_test_signal.pt')
test_dataset = ArielMLDataset(lc_test_path, shuffle=False)
# %%
