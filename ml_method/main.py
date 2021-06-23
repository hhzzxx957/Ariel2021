# %%
from model import LGB
import pandas as pd
from sklearn.model_selection import train_test_split
import gc
import numpy as np
import torch

#%%
# # sample data
# df =pd.DataFrame(torch.load('../data/full_train_signal.pt').numpy())
# indices_tot = list(range(125600))
# np.random.seed(2021)
# np.random.shuffle(indices_tot)

# train_size = 4096*4
# train_ind = indices_tot[:train_size]
# indices = []
# for i in train_ind:
#     indices.extend(list(range(i*55, i*55+55)))

# train_df = df.iloc[indices, :]
# train_df.to_pickle('../data/sample_train_df.pickle')

# %%
train_df = pd.read_pickle('../data/sample_train_df.pickle')
# colnames = list(train_df.columns)
# colnames[-1] = 'label'
# train_df.columns = colnames

y = train_df.pop('300') # label
X = train_df
del train_df
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
gc.collect()

train_size = 4096*4
indices_tot = list(range(train_size))
np.random.shuffle(indices_tot)
train_ind = indices_tot[:int(train_size*0.9)]
val_ind = indices_tot[int(train_size*0.9):]
train_indices = []
for i in train_ind:
    train_indices.extend(list(range(i*55, i*55+55)))
val_indices = []
for i in val_ind:
    val_indices.extend(list(range(i*55, i*55+55)))


X_train, X_valid = X.iloc[train_indices, :], X.iloc[val_indices, :]
y_train, y_valid = y[train_indices], y[val_indices]
# train_test_split(X, y, test_size=0.2, random_state=2021)


#%%
params = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'seed': 2021,
        'num_threads': 20
    }
hyperparams = {
        'two_round': False,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': 9,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'feature_fraction': 0.9,
    }
lgb = LGB(params=params, hyperparams=hyperparams, verbose=1)
valid_pred = lgb.valid_fit(X_train, X_valid, y_train, y_valid,)

# %%
# lgb.model.save_model('model.txt', num_iteration=lgb.model.best_iteration)


#%%
X_test = pd.read_pickle('../data/full_test_df.pickle')

test_pred = lgb.predict(X_test)
train_pred = lgb.predict(X)

# %%
#(53900, 55)
np.savetxt('lgb_test_preds.txt', test_pred.reshape((-1, 55)))
np.savetxt('lgb_train_preds.txt', train_pred.reshape((-1, 55)))
# %%
