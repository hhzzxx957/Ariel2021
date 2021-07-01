# %%
import torch
import numpy as np
import pandas as pd
from constants import *
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft
#%%
full_train_signal_pt = torch.load('data/full_train_signal_preprocessed.pt')
full_train_signal_org_pt = torch.load('data/full_train_signal.pt')
# %%
plt.plot(full_train_signal_org_pt[5609028,:-1])
plt.ylim((0.98, 1.02))
plt.show()
# %%
max_vals = torch.max(full_train_signal_pt[:,:-1], dim=1)[0]
min_vals = torch.min(full_train_signal_pt[:,:-1], dim=1)[0]
# %%
# preds1 = np.loadtxt('outputs/featquan_proprocessed_variance_adamw_cross/ensemble_evaluation_2021-06-29.txt')
# preds1 = np.loadtxt('outputs/featquanphoton_proprocessed_adamw_cross/ensemble_evaluation_2021-06-30.txt')
preds = np.loadtxt('outputs/finalensemble.txt')
print(np.mean(preds))
preds

#%%
preds = (preds1 + preds2)/2
np.savetxt('res/final_ensemble.txt', preds)

# %%
preds[:4000, :] = 0.06
np.savetxt('outputs/featquanlgb_preprocessed_adamw_cross/ensemble_evaluation_2021-06-26_s.txt', preds)
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
file_feat = torch.load('data/file_feat_train.pt')



# %%

# %%
from sklearn.preprocessing import MinMaxScaler
train_feats = pd.read_csv('data/train_properties_df.csv')
test_feats = pd.read_csv('data/test_properties_df.csv')
colnames = train_feats.columns
scaler = MinMaxScaler()
train_feats = scaler.fit_transform(train_feats)
test_feats = scaler.transform(test_feats)

train_feats = pd.DataFrame(train_feats, columns=colnames)
test_feats = pd.DataFrame(test_feats, columns=colnames)

train_feats.to_csv('data/train_properties_norm_df.csv')
test_feats.to_csv('data/test_properties_norm_df.csv')

# %%

def ensemble_predictions(preds_files):
    final_pred = None
    for file in preds_files:
        p = np.loadtxt(file)
        if final_pred is None:
            final_pred = p
        else:
            final_pred += p
    final_pred /= len(preds_files)
    return final_pred

pred_dirs = [
    'outputs/featquanlgb_proprocessed10_adamw_cross/ensemble_evaluation_2021-06-30.txt',
    'outputs/featquanlgb_proprocessed6_adamw_cross/ensemble_evaluation_2021-06-30.txt',
    'outputs/featquanlgb_preprocessed_adamw_cross/ensemble_evaluation_2021-06-26.txt',
    'outputs/featquan_proprocessed10_adamw_cross/ensemble_evaluation_2021-06-30.txt',
    'outputs/featquan_proprocessed6_adamw_cross/ensemble_evaluation_2021-06-30.txt',
    'outputs/featquan_preprocessed_adamw_cross/ensemble_evaluation_2021-06-26.txt',
    'outputs/featquan_fftsmooth_adamw_cross/ensemble_evaluation_2021-06-28.txt',
    'outputs/featquanphoton_proprocessed_adamw_cross/ensemble_evaluation_2021-06-30.txt'
]

filna_preds = ensemble_predictions(pred_dirs)
np.savetxt('outputs/finalensemble.txt', filna_preds)
# %%
from sklearn.manifold import TSNE
preds1 = np.loadtxt('outputs/featquan_preprocessed_adamw_cross/ensemble_evaluation_2021-06-26.txt')
preds2 = np.loadtxt('outputs/featquanlgb_proprocessed10_adamw_cross/ensemble_evaluation_2021-06-30.txt')
preds3 = np.loadtxt('outputs/featquanphoton_proprocessed_adamw_cross/ensemble_evaluation_2021-06-30.txt')

preds_ensem = np.loadtxt('outputs/finalensemble.txt')
# %%
preds1_embedded = TSNE(n_components=2).fit_transform(preds1[:500])
preds2_embedded = TSNE(n_components=2).fit_transform(preds2[:500])
preds3_embedded = TSNE(n_components=2).fit_transform(preds3[:500])

preds_ensem_embedded = TSNE(n_components=2).fit_transform(preds_ensem[:500])
# %%
# initialize a matplotlib plot
plt.scatter(preds1_embedded[:,0],preds1_embedded[:,1])
plt.scatter(preds2_embedded[:,0],preds2_embedded[:,1])
plt.scatter(preds3_embedded[:,0],preds3_embedded[:,1])
plt.scatter(preds_ensem_embedded[:,0],preds_ensem_embedded[:,1])
plt.show()
# %%
np.min(preds_ensem)
# %%
df_denoise = pd.read_pickle('data/full_train_denoise_df.pickle')
planet_id = np.repeat(list(range(1256)), 55)
df_denoise['spec_id'] = planet_id
df_train_mean = df_denoise.groupby(planet_id).mean()
# %%
plt.hist(df_train_mean.iloc[:, 150], bins=20)
plt.xlim(-0.03, 0)
plt.show()
# %%
df_test_denoise = pd.read_pickle('data/full_test_denoise_df.pickle')
planet_id = np.repeat(list(range(539)), 55)
df_test_denoise['spec_id'] = planet_id
df_test_mean = df_test_denoise.groupby(planet_id).mean()
# %%
plt.hist(df_test_mean.iloc[:, 150], bins=20)
plt.xlim(-0.03, 0)
plt.show()
# %%
