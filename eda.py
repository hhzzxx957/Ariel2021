# %%
import torch
import numpy as np
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
preds = np.loadtxt('ml_method/lgb_oof_train_preds.txt')
print(np.mean(preds))
preds

# %%
preds[:2000, :] = 0.06
np.savetxt('outputs/DilatedCNN_kernel3_deep_feat_large_preprocessed_metricadamw_cross/ensemble_evaluation_2021-06-24_s.txt', preds)
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
