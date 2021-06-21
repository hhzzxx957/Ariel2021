# %%
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import ArielMLDataset, ArielMLFeatDataset
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
# preprocess file
from joblib import parallel_backend, Parallel, delayed, effective_n_jobs
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import _num_samples


def parallel_apply(df, func, n_jobs= -1, **kwargs):
    """ Pandas apply in parallel using joblib. 
    Uses sklearn.utils to partition input evenly.
    
    Args:
        df: Pandas DataFrame, Series, or any other object that supports slicing and apply.
        func: Callable to apply
        n_jobs: Desired number of workers. Default value -1 means use all available cores.
        **kwargs: Any additional parameters will be supplied to the apply function
        
    Returns:
        Same as for normal Pandas DataFrame.apply()
        
    """
    
    if effective_n_jobs(n_jobs) == 1:
        return df.apply(func, **kwargs)
    else:
        ret = Parallel(n_jobs=n_jobs)(
            delayed(type(df).apply)(df[s], func, **kwargs)
            for s in gen_even_slices(_num_samples(df), effective_n_jobs(n_jobs)))
        return pd.concat(ret)

def rolling_mean(x):
    return pd.Series(x).rolling(3).mean().fillna(0).values

avg_signal = np.load('/data-nbd/ml_dataset/timeseries/pkdd_ml_data_challenge2/data/average_signal.npy')

mode = 'test'
if mode == 'train':
    full_train_signal_pt = torch.load('data/full_train_signal.pt')
    df = full_train_signal_pt[:,:-1] - avg_signal.astype(np.float32)
    df = pd.DataFrame(df.numpy())
    df_new = parallel_apply(df, rolling_mean, n_jobs=12, axis=1, result_type='expand')
    full_train_signal_pt_new = torch.tensor(df_new.values).type(torch.float32)
    full_train_signal_pt_new = torch.cat((full_train_signal_pt_new, full_train_signal_pt[:,-1].unsqueeze(1)), axis=1)
    torch.save(torch.tensor(full_train_signal_pt_new), 'data/full_train_signal_preprocessed.pt')
elif mode == 'test':
    full_test_signal_pt = torch.load('data/full_test_signal.pt')
    df = full_test_signal_pt - avg_signal.astype(np.float32)
    df = pd.DataFrame(df.numpy())
    df_new = parallel_apply(df, rolling_mean, n_jobs=12, axis=1, result_type='expand')
    full_test_signal_pt_new = torch.tensor(df_new.values).type(torch.float32)
    torch.save(torch.tensor(full_test_signal_pt_new), 'data/full_test_signal_preprocessed.pt')

# %%
