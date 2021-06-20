import lightgbm as lgb
import numpy as np
import optuna
from optuna.integration.lightgbm import LightGBMTuner
import pandas as pd
import copy

class LGB:
    '''
    Customized LightGBM class
    '''
    default_params = {
        'objective': 'binary',
        'verbosity': -1,
        'seed': 20,
        'num_threads': 16
    }
    default_hyperparams = {
        'two_round': False,
        'learning_rate': 0.5,
        'num_leaves': 50,
        'max_depth': 7,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'feature_fraction': 0.9,
        # 'min_sum_hessian_in_leaf': 0.1,
        # 'lambda_l1': 0.5,
        # 'lambda_l2': 0.5,
        # 'min_data_in_leaf': 50,
    }
    simple_hyperparams = {
        'objective': 'binary',
        'n_jobs': 16,
        'metric': 'None',
        'max_depth': 6,
        'eta': 0.015,
        'num_leaves': 40,
        'verbosity': -1
    }

    def __init__(self, params=None, hyperparams=None, verbose=False):
        self.model = None
        self.params = self.default_params
        if params is not None:
            self.params.update(params)
        self.hyperparams = self.default_hyperparams
        if hyperparams is not None:
            self.hyperparams.update(hyperparams)

        self.num_boost_round = 3000
        self.best_iteration = 100
        self.train_size = 0
        self.verbose = verbose

    @staticmethod
    def fname_mapping(feat_name):
        '''
        Get feature name mapping to index
        '''
        new_name = list(map(str, range(0, len(feat_name))))
        idx2fname_maps = {str(i): feat_name[i] for i in range(len(feat_name))}
        fname2idx_maps = {feat_name[i]: str(i) for i in range(len(feat_name))}
        return new_name, idx2fname_maps, fname2idx_maps

    @staticmethod
    def to_numpy(df):
        return df.astype(np.float32).values

    def prepare_data(self,
                     X_train,
                     y_train,
                     X_valid=None,
                     y_valid=None,
                     categorical_feature=None,
                     silent=True):
        '''
        prepare data and info before training
        '''
        self.feature_name = X_train.columns
        new_feat_name, self.idx2fname_maps, fname2idx_maps = self.fname_mapping(
            self.feature_name)
        X_train = self.to_numpy(X_train)
        lgb_train = lgb.Dataset(X_train, label=y_train, silent=silent)

        # get training parameters
        train_params = {
            'train_set': lgb_train,
            'feature_name': new_feat_name,
            'num_boost_round': self.num_boost_round,
            'verbose_eval': 50
        }

        # input categorical features
        if categorical_feature is not None:
            categorical_feature = [
                fname2idx_maps[i] for i in categorical_feature
            ]
            train_params['categorical_feature'] = categorical_feature

        # when validation set exists
        if X_valid is not None and y_valid is not None:
            X_valid = self.to_numpy(X_valid)
            lgb_eval = lgb.Dataset(X_valid, label=y_valid, \
                            reference=lgb_train, silent=silent)
            train_params = {
                'train_set': lgb_train,
                'valid_sets': [lgb_train, lgb_eval],
                'feature_name': new_feat_name,
                'valid_names': ['train', 'eval'],
                'num_boost_round': self.num_boost_round,
                'early_stopping_rounds': 30,
                'verbose_eval': 50
            }
        return train_params

    def run_cv(self, X_train, y_train, cv_params):
        origin_feat_name = X_train.columns
        new_feat_name, _, fname2idx_maps = self.fname_mapping(origin_feat_name)
        dtrain = lgb.Dataset(X_train, y_train, free_raw_data=True, silent=True)
        categorical_feature = [
            fname2idx_maps[i] for i in cv_params['categorical_feature']
        ]
        cv_params.update({
            'train_set': dtrain,
            'feature_name': new_feat_name,
            'categorical_feature': categorical_feature,
        })
        hist = lgb.cv(**cv_params)
        return hist

    def run_simple_model(self,
                         X_train,
                         y_train,
                         params=None,
                         categorical_feature=None):
        train_params = self.prepare_data(X_train, y_train, \
                        categorical_feature=categorical_feature, silent=True)
        origin_feat_name = self.feature_name
        if params is None:
            params = self.simple_hyperparams
        train_params['num_boost_round'] = 50
        model = lgb.train(params, **train_params)
        return model, origin_feat_name

    def valid_fit(self,
                  X_train,
                  X_valid,
                  y_train,
                  y_valid,
                  categorical_feature=None):
        train_params = self.prepare_data(
            X_train,
            y_train,
            X_valid,
            y_valid,
            categorical_feature=categorical_feature)
        if not self.verbose:
            train_params['verbose_eval'] = 0
        self.model = lgb.train({
            **self.params,
            **self.hyperparams
        }, **train_params)
        self.best_iteration = self.model.best_iteration
        # self.model.best_score["eval"][self.params["metric"]]
        return self.model.predict(X_valid)

    def fit(self,
            X_train,
            y_train,
            categorical_feature=None,
            additional_iterations_ratio=0.07):

        train_params = self.prepare_data(
            X_train, y_train, categorical_feature=categorical_feature)
        best_iteration = int(
            (1 + additional_iterations_ratio) * self.best_iteration)
        train_params['num_boost_round'] = best_iteration
        if not self.verbose:
            train_params['verbose_eval'] = 0

        self.model = lgb.train({
            **self.params,
            **self.hyperparams
        }, **train_params)
        return self.model.predict(X_train)

    def predict(self, df):
        df = df[self.feature_name]
        return self.model.predict(df)

    def hyparam_opt(self,
                    X_train,
                    X_valid,
                    y_train,
                    y_valid,
                    categorical_feature=None,
                    time_budget=300):
        '''Optimize hyperparameters of LGB

        Wrapper of optuna LightGBMTuner
        '''
        train_params = self.prepare_data(
            X_train,
            y_train,
            X_valid,
            y_valid,
            categorical_feature=categorical_feature)
        train_params['verbose_eval'] = 0
        optuna.logging.disable_default_handler()
        optimizer = LightGBMTuner({
            **self.params,
            **self.hyperparams
        },
                                  **train_params,
                                  show_progress_bar=self.verbose,
                                  time_budget=time_budget)
        optimizer.run()
        self.hyperparams.update(optimizer.best_params)
        # return optimizer.get_best_booster()
    
    def update_params(self, params):
        self.hyperparams.update(params)

    def feat_imp(self):
        '''
        get feature importance
        '''
        df_imp = pd.DataFrame({
            'features': [i for i in self.feature_name],
            'importances':
            self.model.feature_importance('gain')
        })

        df_imp.sort_values('importances', ascending=False, inplace=True)
        return df_imp

    def ensemble_train(self, X_train, y_train, categorical_feature=None, \
            additional_iterations_ratio=0.07, model_num=10):
        '''Do bagging for LightGBM.

        Train several times, add some randomness to parameters in every iteration.
        | Use this function after valid_fit to get best iterations

        Parameters
        ----------
        X_train: pd.DataFrame
        y_train: pd.Series
        categorical_feature: List(str)
        additonal_iterations_ratio: float
            additional number of iterations used in fitting
        model_num: int
            model number

        Returns
        ----------
        ensemble_models: List(object)
            list of models
        '''
        self.ensemble_models = []
        self.ensemble_columns = []

        self.feature_name = X_train.columns
        new_feat_name, self.idx2fname_maps, self.fname2idx_maps = \
            self.fname_mapping(self.feature_name)
        X_train = self.to_numpy(X_train)
        lgb_train = lgb.Dataset(X_train, label=y_train)
        best_iteration = int(
            (1 + additional_iterations_ratio) * self.best_iteration)
        train_params = {
            'train_set': lgb_train,
            'valid_sets': lgb_train,
            'feature_name': new_feat_name,
            'valid_names': 'train',
            'num_boost_round': best_iteration,
            'early_stopping_rounds': 30,
            'verbose_eval': 50,
        }
        if categorical_feature is not None:
            categorical_feature = [
                self.fname2idx_maps[i] for i in categorical_feature
            ]
            train_params['categorical_feature'] = categorical_feature
        if not self.verbose:
            train_params['verbose_eval'] = 0

        model = lgb.train({**self.params, **self.hyperparams}, **train_params)

        self.ensemble_models.append(model)
        self.ensemble_columns.append(new_feat_name)

        for i in range(1, model_num + 1):

            seed = np.random.randint(2020 * i, 2020 * (i + 1))

            cur_columns = list(
                pd.Series(new_feat_name).sample(frac=0.85,
                                                replace=False,
                                                random_state=seed))
            X_sample = X_train[:, [int(col) for col in cur_columns]]

            # params
            params = self.params
            params['seed'] = seed

            # hyperparams
            cur_hyperparams = copy.deepcopy(self.hyperparams)
            num_leaves = self.hyperparams['num_leaves']
            cur_hyperparams['num_leaves'] = num_leaves + \
                np.random.randint(-int(num_leaves/10),int(num_leaves/10)+7)

            cur_iteration = self.best_iteration
            cur_iteration = cur_iteration + np.random.randint(-15, 30)

            if cur_iteration <= 20:
                cur_iteration = self.best_iteration

            train_data = lgb.Dataset(X_sample,
                                     label=y_train,
                                     feature_name=cur_columns)

            model = lgb.train({
                **params,
                **cur_hyperparams
            },
                              train_data,
                              num_boost_round=cur_iteration,
                              feature_name=cur_columns)

            self.ensemble_columns.append(cur_columns)
            self.ensemble_models.append(model)

        return self.ensemble_models

    def ensemble_predict(self, X_test):
        '''
        do prediction on test set. use after ensemble_train

        Returns
        ----------
        preds: pd.Series
            averaged prediction
        '''
        assert self.ensemble_models is not None
        X_t = X_test.copy()
        new_names = [self.fname2idx_maps[col] for col in X_t.columns]
        X_t.columns = new_names
        # pred_tot = np.zeros(X_t.shape[0]) # not suitable for multiclass
        pred_tot = None
        for i, model in enumerate(self.ensemble_models):
            if pred_tot is None:
                pred_tot = model.predict(X_t[self.ensemble_columns[i]])
            else:
                pred_tot += model.predict(X_t[self.ensemble_columns[i]])
        preds = pred_tot / len(self.ensemble_models)
        return preds
