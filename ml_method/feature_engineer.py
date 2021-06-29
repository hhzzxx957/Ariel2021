# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

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
aggregate_df(save_name='../data/full_test_denoise_df.pickle', mode='eval')

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
def aggregate_df_gaussavg(save_name, mode='train'):
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
aggregate_df(save_name='../data/full_test_denoise_df.pickle', mode='eval')


#%%
file_feat = get_file_feats(lc_train_path, 'data/file_feat_train.pt')