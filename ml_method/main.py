# %%
from model import LGB
import pandas as pd
from sklearn.model_selection import train_test_split
import gc
import numpy as np
import torch
from sklearn.model_selection import KFold

def scoring(pred, y, loss=False):
    if loss:
        return (y * np.abs(pred - y)).sum() / len(y) * 1e6
    else:
        return 1e4 - 2 * (y * np.abs(pred - y)).sum() / len(y) * 1e6

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
def train_lgb(train_df, X_test):
    # train_df = pd.read_pickle('../data/full_train_df.pickle')
    # colnames = list(train_df.columns)
    # colnames[-1] = 'label'
    # train_df.columns = colnames

    y = train_df.pop('label') # label
    X = train_df
    del train_df
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    # X_test = pd.read_pickle('../data/full_test_df.pickle')
    gc.collect()

    train_size = 1256 #4096*4
    step = 5500
    indices_tot = list(range(train_size))
    # np.random.shuffle(indices_tot)
    # train_ind = indices_tot[:int(train_size*0.9)]
    # valid_ind = indices_tot[int(train_size*0.9):]
    final_pred = np.zeros(len(y))
    final_test_pred = np.zeros(len(X_test))
    kf = KFold(n_splits=5, shuffle=True, random_state=2021)
    for i, (train_ind, valid_ind) in enumerate(kf.split(indices_tot)):
        train_indices = []
        for i in train_ind:
            train_indices.extend(list(range(i*step, (i+1)*step)))
        val_indices = []
        for i in valid_ind:
            val_indices.extend(list(range(i*step, (i+1)*step)))


        X_train, X_valid = X.iloc[train_indices, :], X.iloc[val_indices, :]
        y_train, y_valid = y[train_indices], y[val_indices]
    # train_test_split(X, y, test_size=0.2, random_state=2021)

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
                'max_depth': 8,
                'bagging_fraction': 0.9,
                'bagging_freq': 5,
                'feature_fraction': 0.9,
            }
        lgb = LGB(params=params, hyperparams=hyperparams, verbose=1)
        valid_pred = lgb.valid_fit(X_train, X_valid, y_train, y_valid,)
        test_pred = lgb.predict(X_test)
        print(scoring(valid_pred, y_valid))
        print(scoring(valid_pred, y_valid, loss=True))
        
        final_pred[val_indices] = valid_pred
        final_test_pred += test_pred
    final_test_pred /= 5
    print(lgb.feat_imp()[:20])
    np.savetxt('lgb_oof_train_preds.txt', final_pred.reshape((-1, 55)))
    np.savetxt('lgb_oof_test_preds.txt', final_test_pred.reshape((-1, 55)))
# %%
# lgb.model.save_model('model.txt', num_iteration=lgb.model.best_iteration)
# test_pred = lgb.predict(X_test)
# lgb.feat_imp()


# %%
if __name__ == "__main__":
    train_df = pd.read_pickle('../data/fullfeat_train_df.pickle')
    X_test = pd.read_pickle('../data/fullfeat_test_df.pickle')
    train_lgb(train_df, X_test)
# %%
