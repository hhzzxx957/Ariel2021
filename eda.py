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
preds1 = np.loadtxt('outputs/featquan_preprocessed_adamw_cross/ensemble_evaluation_2021-06-26.txt')
preds2 = np.loadtxt('outputs/featquanlgb_preprocessed_adamw_cross/ensemble_evaluation_2021-06-26.txt')
# preds = np.loadtxt('ml_method/lgb_oof_test_preds.txt')
print(np.mean(preds1))
preds1

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
