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
def get_signal_average(save_name, mode='mean'):
    # dataset = ArielMLDataset(lc_train_path, params_train_path, shuffle=False)
    # output = np.zeros(dataset[0]['lc'].T.detach().numpy().shape)
    # tot_len = len(dataset)
    # for i in tqdm(range(tot_len)):
    #     output += dataset[i]['lc'].T.detach().numpy()
    # output /= tot_len
    output = torch.load('data/full_train_signal.pt')
    print(output.shape)

    if mode == 'mean':
        res = torch.mean(output[:,:-1], dim=0)[0].numpy()
    else:
        res = torch.median(output[:,:-1], dim=0)[0].numpy()

    # res[20:] = 1
    print(res.shape)
    np.save(save_name, res)
get_signal_average('data/median_signal.npy', mode='median')


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
    return pd.Series(x).rolling(6).mean().fillna(0).values

def fft_smooth(x):
    signal_fft = rfft(x)
    signal_fft[20:] = 0
    return irfft(signal_fft)

def get_fft(x):
    signal_fft = np.abs(rfft(x))
    return signal_fft

avg_signal = np.load('data/average_signal.npy')

def preprocess_signal(full_data_dir, save_name, mode='train', add_label=True):
    full_data = torch.load(full_data_dir)
    if mode == 'train':
        df = full_data[:, :-1] - avg_signal.astype(np.float32)
    elif mode == 'eval':
        df = full_data - avg_signal.astype(np.float32)
    df = pd.DataFrame(df.numpy())
    print(df.shape)
    df_new = parallel_apply(df,
                            rolling_mean,
                            n_jobs=15,
                            axis=1,
                            result_type='expand')
    # df_new = df
    print('apply done')
    full_data_new = torch.tensor(df_new.values).type(torch.float32)
    if mode == 'train' and add_label:
        full_data_new = torch.cat(
            (full_data_new, full_data[:, -1].unsqueeze(1)), axis=1)
    torch.save(full_data_new, save_name)

preprocess_signal('data/full_train_signal.pt',
                  save_name='data/full_train_signal_preprocess6.pt',
                  mode='train',
                  add_label=True
                  )
preprocess_signal('data/full_test_signal.pt',
                  save_name='data/full_test_signal_preprocess6.pt',
                  mode='eval')


# %%
# generate full data
def generate_full_dataframe(mode='train'):
    # configs
    if mode == 'train':
        start_ind, end_ind = 0, 125600
        tot_len = 125600
    else:
        start_ind, end_ind = 125600, 125600+53900
        tot_len = 53900

    main_table_dir = f'../data/full_{mode}_signal.pt'
    file_feat_dir = f'../data/file_feat_{mode}.pt'
    feat_dir = f'../data/{mode}_properties_df.csv'

    # prepare main table
    full_signal = torch.load(main_table_dir)
    df = pd.DataFrame(full_signal.numpy())

    colnames = list(df.columns)
    if mode == 'train':
        colnames[-1] = 'label'
        df.columns = colnames

    ids = []
    for i in range(start_ind, end_ind):
        ids.extend([i]*55)

    positions = list(range(55))*tot_len

    df['id'] = ids
    df['position'] = positions

    print('load main table done')

    # prepare feature tables
    file_feat = pd.DataFrame(torch.load(file_feat_dir).numpy())
    feat_df = pd.read_csv(feat_dir)

    #concat
    feat_df = pd.concat([feat_df, file_feat], axis=1)

    # merge
    feat_df['id'] = list(range(start_ind, end_ind))
    df_all = pd.merge(df, feat_df, on='id')

    print('merge table done')

    df_all.to_pickle(f'../data/full_{mode}_df.pickle')

# generate_full_dataframe(mode='train')
generate_full_dataframe(mode='test')
# %%
# %%
# noise aggregate
def aggregate_df(save_name, mode='train'):
    if mode == 'train':
        tot_len = 125600
        full_signal = torch.load('../data/full_train_signal.pt')
    else:
        tot_len = 53900
        full_signal = torch.load('../data/full_test_signal.pt')    

    avg = np.load('../data/average_signal_new.npy')

    if mode == 'train':
        full_df = pd.DataFrame(full_signal[:,:-1].numpy())
    else:
        full_df = pd.DataFrame(full_signal.numpy())

    spec_id = list(range(55))*tot_len
    planet_id = []
    for i in range(tot_len//100):
        planet_id.extend([i]*100*55)
    
    full_df['spec_id'] = spec_id
    full_df['planet_id'] = planet_id
    
    full_denoise_df = full_df.groupby(['planet_id', 'spec_id']).median()
    full_denoise_df -= avg
    
    full_denoise_df.to_pickle(save_name)

# aggregate_df(save_name='../data/full_train_denoise_df.pickle', mode='train')
aggregate_df(save_name='data/full_test_denoise_df.pickle', mode='eval')

# %%
def get_quantile_feat(save_name, mode='train'):
    if mode == 'train':
        full_denoise_df = pd.read_pickle('../data/full_train_denoise_df.pickle')
    else:
        full_denoise_df = pd.read_pickle('../data/full_test_denoise_df.pickle')

    quantile_01 = np.sqrt(-full_denoise_df.quantile(0.1, axis=1).values).reshape(-1,1)
    quantile_001 = np.sqrt(-full_denoise_df.quantile(0.05, axis=1).values).reshape(-1,1)
    quantile_005 = np.sqrt(-full_denoise_df.quantile(0.01, axis=1).values).reshape(-1,1)
    quantile_feat = np.concatenate([quantile_01, quantile_005, quantile_001], axis=1)
    quantile_feat = quantile_feat.astype(np.float32)

    np.save(save_name, quantile_feat)

# get_quantile_feat('../data/quantile_feat_train.npy', mode='train')
get_quantile_feat('../data/quantile_feat_test.npy', mode='eval')
# %%
def merge_quantile_feat(save_name, mode='train'):
    if mode == 'train':
        quantile_df = pd.DataFrame(np.load('../data/quantile_feat_train.npy'))
        full_df = pd.read_pickle('../data/full_train_df.pickle')
    else:
        quantile_df = pd.DataFrame(np.load('../data/quantile_feat_test.npy'))
        full_df = pd.read_pickle('../data/full_test_df.pickle')

    tot_len = len(quantile_df)//55
    plant_ids_1 = np.repeat(list(range(tot_len)), 55*100)
    plant_ids_2 = np.repeat(list(range(tot_len)), 55)
    spec_id = list(range(55))*tot_len

    quantile_df['planet_id'] = plant_ids_2
    quantile_df['position'] = spec_id
    full_df['planet_id'] = plant_ids_1

    full_df = pd.merge(full_df, quantile_df, on=['planet_id', 'position'])

    full_df.to_pickle(save_name)

# merge_quantile_feat(save_name='../data/fullfeat_train_df.pickle', mode='train')
merge_quantile_feat(save_name='../data/fullfeat_test_df.pickle', mode='test')
# %%
def aggregate_df_photonavg(save_name, mode='train'):
    if mode == 'train':
        tot_len = 125600
        full_signal = torch.load('data/full_train_signal.pt')
        file_feat = torch.load('data/file_feat_train.pt')
    else:
        tot_len = 53900
        full_signal = torch.load('data/full_test_signal.pt')    
        file_feat = torch.load('data/file_feat_test.pt')

    avg = np.load('data/average_signal_new.npy')

    if mode == 'train':
        full_df = pd.DataFrame(full_signal[:,:-1].numpy())
    else:
        full_df = pd.DataFrame(full_signal.numpy())

    spec_id = list(range(55))*tot_len
    planet_id = []
    for i in range(tot_len//100):
        planet_id.extend([i]*100*55)

    photon_id = []
    photon_ids = list(file_feat[:, 2].numpy())
    for i in photon_ids:
        photon_id.extend([i]*55)
    
    full_df['spec_id'] = spec_id
    full_df['planet_id'] = planet_id
    full_df['photon_id'] = photon_id
    
    full_denoise_df = full_df.groupby(['planet_id', 'spec_id','photon_id']).median()
    full_denoise_df -= avg
    
    full_denoise_df.to_pickle(save_name)

# aggregate_df_photonavg(save_name='data/full_train_denoisephoton_df.pickle', mode='train')
aggregate_df_photonavg(save_name='data/full_test_denoisephoton_df.pickle', mode='eval')

# %%
def get_quantile_featphoton(save_name, mode='train'):
    if mode == 'train':
        full_denoise_df = pd.read_pickle('data/full_train_denoisephoton_df.pickle')
    else:
        full_denoise_df = pd.read_pickle('data/full_test_denoisephoton_df.pickle')

    quantile_01 = np.sqrt(-full_denoise_df.quantile(0.1, axis=1).values).reshape(-1,1)
    quantile_001 = np.sqrt(-full_denoise_df.quantile(0.05, axis=1).values).reshape(-1,1)
    quantile_005 = np.sqrt(-full_denoise_df.quantile(0.01, axis=1).values).reshape(-1,1)
    quantile_feat = np.concatenate([quantile_01, quantile_005, quantile_001], axis=1)
    quantile_feat = quantile_feat.astype(np.float32)

    np.save(save_name, quantile_feat)
get_quantile_featphoton('data/quantile_featphoton_train.npy', mode='train')
get_quantile_featphoton('data/quantile_featphoton_test.npy', mode='eval')
# %%
