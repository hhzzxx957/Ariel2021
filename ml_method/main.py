# %%
from model import LGB
import pandas as pd
from sklearn.model_selection import train_test_split
import gc
import numpy as np


# %%
train_df = pd.read_pickle('../data/full_train_df.pickle')

y = train_df.pop('label')
X = train_df
del train_df
gc.collect()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=2021)
del X, y
gc.collect()


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
        'learning_rate': 0.5,
        'num_leaves': 31,
        'max_depth': 9,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'feature_fraction': 0.9,
    }
lgb = LGB(params=params, hyperparams=hyperparams, verbose=1)
valid_pred = lgb.valid_fit(X_train, X_valid, y_train, y_valid,)




#%%
X_test = pd.read_pickle('../data/full_test_df.pickle')

test_pred = lgb.predict(X_test)


# %%
#(53900, 55)
np.savetxt('lgb.txt', test_pred.reshape((-1, 55)))
# %%
