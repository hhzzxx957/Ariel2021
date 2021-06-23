# %%
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import ArielMLDataset, ArielMLFeatDataset
from joblib import Parallel, delayed
import torch
import gc
import re
from constants import *

data_dir = "/data-nbd/ml_dataset/timeseries/pkdd_ml_data_challenge2"

#%%
def get_file_feats(file_dir, save_name):
    files = sorted([p for p in os.listdir(file_dir) if p.endswith('txt')])
    def extract_file_feat(i):
        # return re.split('_|\.', files[i])[:3]
        file_feat = files[i].split('.')[0].split('_')
        return [int(n) for n in file_feat]
    output = Parallel(n_jobs=1)(delayed(extract_file_feat)(i) for i in tqdm(range(len(files))))
    file_feats = torch.tensor(output)
    torch.save(file_feats, save_name)
get_file_feats(lc_train_path, 'data/file_feat_train.pt')
get_file_feats(lc_test_path, 'data/file_feat_test.pt')

#%%
def get_features(file_dir, save_name):
    files = sorted([p for p in os.listdir(file_dir) if p.endswith('txt')])

    output = []
    for file in tqdm(files):
        with open(os.path.join(file_dir,file), 'r') as f:
            properties = f.readlines()[:6]
            res = []
            for s in properties:
                res.append(float(s.split(': ')[1]))
            output.append(res)

    df = pd.DataFrame(output, columns=['star_temp', 'star_logg', 'star_rad', 'star_mass', 'star_k_mag', 'period'])
    df.to_csv(save_name, drop_index=True)

get_features(lc_train_path, 'data/properties_df.csv')
get_features(lc_test_path, 'data/test_properties_df.csv')

# %%
# get average
def get_signal_average(save_name):
    dataset = ArielMLDataset(lc_train_path, params_train_path, shuffle=False)
    output = np.zeros(dataset[0]['lc'].T.detach().numpy().shape)
    tot_len = len(dataset)
    for i in tqdm(range(tot_len)):
        output += dataset[i]['lc'].T.detach().numpy()
    output /= tot_len
    res = np.mean(output, axis=1)

    # res[20:] = 1
    print(res.shape)
    np.save(save_name, res)
get_signal_average('data/average_signal_new.npy')

#%%
# aggregate file
def aggregate_files(save_name, mode='train'):
    if mode == 'train':
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

    elif mode == 'eval':
        dataset = ArielMLDataset(lc_test_path, shuffle=False)
        tot_len = len(dataset)
        def extract_signal(i):
            return dataset[i]['lc'].detach().numpy()

        output = Parallel(n_jobs=20)(delayed(extract_signal)(i) for i in tqdm(range(tot_len)))

    res = np.concatenate(output, axis=0)
    print(res.shape)
    torch.save(torch.tensor(res).type(torch.float32), save_name)
    # np.save(save_name, res)
aggregate_files('data/full_train_signal.pt', mode='train')
aggregate_files('data/full_test_signal.pt', mode='eval')
#%%


# %%
# generate full data
def generate_full_dataframe():
    tot_len = 53900
    ids = []
    for i in range(125600, 125600+53900):
        ids.extend([i]*55)

    positions = list(range(55))*(tot_len)
    print(len(ids), len(positions))

    res = torch.load('data/full_test_signal.pt')
    df = pd.DataFrame(res.numpy())
    df['id'] = ids
    df['position'] = positions

    colnames = list(df.columns)
    # colnames[-3] = 'label'
    df.columns = colnames

    # feat_df = pd.read_csv('data/properties_df.csv')
    feat_df = pd.read_csv('data/test_properties_df.csv')

    feat_df['id'] = list(range(125600, 125600+53900)) # 125600
    feat_df.drop('Unnamed: 0', axis=1, inplace=True)
    df_all = pd.merge(df, feat_df, on='id')

    df_all.to_pickle('data/full_test_df.pickle')

# %%
# get average std
def get_average_std(signal):
    # avg = np.load('data/average_signal_new.npy')

    sub_signal = signal[:,:300] #-avg
    std_signal = np.std(sub_signal.numpy(), axis=1)
    print('mean std', np.mean(std_signal))
    print('median std', np.median(std_signal))
signal = torch.load('data/full_train_signal_fftsmooth.pt')
get_average_std(signal)
# %%
# preprocess file
from joblib import parallel_backend, Parallel, delayed, effective_n_jobs
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import _num_samples
# from scipy.fft import fft, ifft, rfft, irfft
from numpy.fft import rfft, irfft

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

def fft_smooth(x):
    signal_fft = rfft(x)
    signal_fft[20:] = 0
    return irfft(signal_fft)


avg_signal = np.load('data/average_signal.npy')

def preprocess_signal(full_data_dir, save_name, mode='train'):
    full_data = torch.load(full_data_dir)
    if mode == 'train':
        df = full_data[:, :-1] - avg_signal.astype(np.float32)
    elif mode == 'eval':
        df = full_data - avg_signal.astype(np.float32)
    df = pd.DataFrame(df.numpy())
    print(df.shape)
    df_new = parallel_apply(df,
                            fft_smooth,
                            n_jobs=15,
                            axis=1,
                            result_type='expand')
    print('apply done')
    full_data_new = torch.tensor(df_new.values).type(torch.float32)
    if mode == 'train':
        full_data_new = torch.cat(
            (full_data_new, full_data[:, -1].unsqueeze(1)), axis=1)
    torch.save(full_data_new, save_name)


# preprocess_signal('data/full_train_signal.pt',
#                   save_name='data/full_train_signal_fftsmooth.pt',
#                   mode='train')
preprocess_signal('data/full_test_signal.pt',
                  save_name='data/full_test_signal_fftsmooth.pt',
                  mode='eval')
# %%
import matplotlib.pyplot as plt
full_train_signal_pt = torch.load('data/full_train_signal_preprocessed.pt')
# %%
plt.plot(full_train_signal_org_pt[5609028,:-1])
plt.ylim((0.98, 1.02))
plt.show()
# %%
max_vals = torch.max(full_train_signal_pt[:,:-1], dim=1)[0]
min_vals = torch.min(full_train_signal_pt[:,:-1], dim=1)[0]
# %%
preds = np.loadtxt('outputs/DilatedCNN_kernel3_deep_featlgb_large_preprocessed_metricadamw/evaluation_2021-06-22.txt')
print(np.mean(preds))
preds

# %%
preds[:2000, :] = 0.06
np.savetxt('outputs/DilatedCNN_kernel3_deep_feat_large_preprocessed_metricadamw/evaluation_2021-06-22_s.txt', preds)
# %%
preds_lgb = np.loadtxt(lgb_test_path)
pres_train_lgb = np.loadtxt(lgb_train_path)
# %%
avg_signal = np.load('data/average_signal_new.npy')

ind = 1
signal_fft = rfft(full_train_signal_org_pt[ind,:-1].numpy()-avg_signal)

# plt.plot(np.abs(signal_fft))

plt.plot(full_train_signal_org_pt[ind,:-1].numpy()-avg_signal)
signal_fft[20:] = 0
plt.plot(irfft(signal_fft))

# %%

