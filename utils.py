"""Define generic classes and functions to facilitate baseline construction"""
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import Module, Sequential
from torch.utils.data import Dataset

from constants import *


class ArielMLDataset(Dataset):
    """Class for reading files for the Ariel ML data challenge 2021"""
    def __init__(self,
                 lc_path,
                 params_path=None,
                 transform=None,
                 start_ind=0,
                 max_size=int(1e9),
                 shuffle=True,
                 seed=None,
                 device=None):
        """Create a pytorch dataset to read files for the Ariel ML Data challenge 2021

        Args:
            lc_path: str
                path to the folder containing the light curves files
            params_path: str
                path to the folder containing the target transit depths (optional)
            transform: callable
                transformation to apply to the input light curves
            start_ind: int
                where to start reading the files from (after ordering)
            max_size: int
                maximum dataset size
            shuffle: bool
                whether to shuffle the dataset order or not
            seed: int
                numpy seed to set in case of shuffling
            device: str
                torch device
        """
        self.lc_path = lc_path
        self.transform = transform
        self.device = device

        self.files = sorted(
            [p for p in os.listdir(self.lc_path) if p.endswith('txt')])
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.files)
        self.files = self.files[start_ind:start_ind + max_size]

        if params_path is not None:
            self.params_path = params_path
        else:
            self.params_path = None
            self.params_files = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item_lc_path = Path(self.lc_path) / self.files[idx]
        lc = torch.from_numpy(np.loadtxt(item_lc_path)).type(torch.float32)
        if self.transform:
            lc = self.transform(lc)
        if self.params_path is not None:
            item_params_path = Path(self.params_path) / self.files[idx]
            target = torch.from_numpy(np.loadtxt(item_params_path)).type(
                torch.float32)
        else:
            target = torch.Tensor()
        return {'lc': lc.to(self.device), 'target': target.to(self.device)}


class ArielMLFeatDataset(Dataset):
    """Class for reading files for the Ariel ML data challenge 2021"""
    def __init__(self,
                 lc_path,
                 feat_path=None,
                 lgb_feat_path=None,
                 file_feat_path=None,
                 quantile_feat_path=None,
                #  quantilephoton_feat_path=None,
                 transform=None,
                 sample_ind=None,
                 device=None,
                 mode='train',
                 ):
        """Create a pytorch dataset to read files for the Ariel ML Data challenge 2021

        Args:
            lc_path: str
                path to the folder containing the light curves files
            params_path: str
                path to the folder containing the target transit depths (optional)
            transform: callable
                transformation to apply to the input light curves
            start_ind: int
                where to start reading the files from (after ordering)
            max_size: int
                maximum dataset size
            shuffle: bool
                whether to shuffle the dataset order or not
            seed: int
                numpy seed to set in case of shuffling
            device: str
                torch device
        """
        self.lc_path = lc_path
        self.transform = transform
        self.device = device
        self.sample_ind = sample_ind
        self.mode = mode

        self.avg_vals = np.load(avg_val_path)

        indices = []
        for i in sample_ind:
            indices.extend(list(range(i*55, i*55+55)))
        quantile_indices = []
        for i in sample_ind:
            quantile_indices.extend(list(range(i//100*55, i//100*55+55)))
        quantilephoton_indices = []
        for i in sample_ind:
            quantilephoton_indices.extend(list(range(i//10*55, i//10*55+55)))

        self.data = torch.load(lc_path)[indices]

        if self.mode != 'eval':
            self.target = self.data[:, -1]
            self.data = self.data[:, :-1]
        self.feats = torch.from_numpy(pd.read_csv(feat_path).values)[sample_ind]
        if file_feat_path is not None:
            feat_file = torch.load(file_feat_path)[sample_ind]
            self.feats = torch.cat([self.feats, feat_file], axis=1)

        if quantile_feat_path is not None:
            quantile_feat = torch.from_numpy(np.load(quantile_feat_path))[quantile_indices]
            quantile_feat = quantile_feat.reshape(-1, 55*3)
            # print(quantile_feat.shape)
            self.feats = torch.cat([self.feats, quantile_feat], axis=1)

        # if quantilephoton_feat_path is not None:
        #     quantilephoton_feat = torch.from_numpy(np.load(quantilephoton_feat_path))[quantilephoton_indices]
        #     quantilephoton_feat = quantilephoton_feat.reshape(-1, 55*3)
        #     # print(quantilephoton_feat.shape)
        #     self.feats = torch.cat([self.feats, quantilephoton_feat], axis=1)

        if lgb_feat_path is not None:
            feat_lgb = torch.from_numpy(np.loadtxt(lgb_feat_path))[sample_ind]
            self.feats = torch.cat([self.feats, feat_lgb], axis=1)

    def __len__(self):
        return len(self.sample_ind)

    def __getitem__(self, idx):
        lc = self.data[55*idx:55*idx+55]

        if self.transform:
            lc = self.transform(lc, self.avg_vals) #self.avg_vals
        feat = self.feats[idx].type(torch.float32)
        if self.mode != 'eval':
            target = self.target[55*idx:55*idx+55]
        else:
            target = torch.Tensor()
        return {
            'lc': lc.to(self.device),
            'target': target.to(self.device),
            'feat': feat.to(self.device),
        }


def simple_transform(x, avg_vals):
    """Perform a simple preprocessing of the input light curve array
    Args:
        x: np.array
            first dimension is time, at least 30 timesteps
    Return:
        preprocessed array
    """
    out = x.clone()
    # centering
    out -= 1.
    # rough rescaling
    out /= 0.04
    return out

def subavg_transform(x, avg_vals):
    out = x.clone()
    # out = torch.clip(out, min=-0.1, max=0.1)
    # out -= avg_vals.astype(np.float32)
    out /= 0.02 #0.006
    return out

def generate_indice(indices, step=100):
    output_indices = []
    for i in indices:
        output_indices.extend(list(range(i*step, (i+1)*step)))
    return output_indices

class ChallengeMetric:
    """Class for challenge metric"""
    def __init__(self, weights=None):
        """Create a callable object close to the Challenge's metric score

        __call__ method returns the error and score method returns the unweighted challenge metric

        Args:
            weights: iterable
                iterable containing the weights for each observation point (default None will create unity weights)
        """
        self.weights = weights

    def __call__(self, y, pred):
        """Return the unweighted error related to the challenge, as defined (here)[https://www.ariel-datachallenge.space/ML/documentation/scoring]

        Args:
            y: torch.Tensor
                target tensor
            pred: torch.Tensor
                prediction tensor, same shape as y
        Return: torch.tensor
            error tensor (itemisable), min value = 0
        """
        y = y
        pred = pred
        if self.weights is None:
            weights = torch.ones_like(y, requires_grad=False)
        else:
            weights = self.weights

        return (weights * y * torch.abs(pred - y)).sum() / weights.sum() * 1e6

    def score(self, y, pred):
        """Return the unweighted score related to the challenge, as defined (here)[https://www.ariel-datachallenge.space/ML/documentation/scoring]

        Args:
            y: torch.Tensor
                target tensor
            pred: torch.Tensor
                prediction tensor, same shape as y
        Return: torch.tensor
            score tensor (itemisable), max value = 10000
        """
        y = y
        pred = pred
        if self.weights is None:
            weights = torch.ones_like(y, requires_grad=False)
        else:
            weights = self.weights

        return (
            1e4 - 2 *
            (weights * y * torch.abs(pred - y)).sum() / weights.sum() * 1e6)


class Baseline(Module):
    """Baseline model for Ariel ML data challenge 2021"""
    def __init__(self,
                 H1=1024,
                 H2=256,
                 input_dim=n_wavelengths * n_timesteps,
                 output_dim=n_wavelengths):
        """Define the baseline model for the Ariel data challenge 2021

        Args:
            H1: int
                first hidden dimension (default=1024)
            H2: int
                second hidden dimension (default=256)
            input_dim: int
                input dimension (default = 55*300)
            ourput_dim: int
                output dimension (default = 55)
        """
        super().__init__()
        self.network = Sequential(
            torch.nn.Linear(input_dim, H1),
            torch.nn.ReLU(),
            torch.nn.Linear(H1, H2),
            torch.nn.ReLU(),
            torch.nn.Linear(H2, output_dim),
        )

    def __call__(self, x):
        """Predict rp/rs from input tensor light curve x"""
        out = torch.flatten(
            x, start_dim=1
        )  # Need to flatten out the input light curves for this type network
        out = self.network(out)
        return out


def cosine(epoch, t_max, ampl):
    """Shifted and scaled cosine function."""

    t = epoch % t_max
    return (1 + np.cos(np.pi * t / t_max)) * ampl / 2


def inv_cosine(epoch, t_max, ampl):
    """A cosine function reflected on X-axis."""

    return 1 - cosine(epoch, t_max, ampl)


def one_cycle(epoch, t_max, a1=0.6, a2=1.0, pivot=0.3):
    """A combined schedule with two cosine half-waves."""

    pct = epoch / t_max
    if pct < pivot:
        return inv_cosine(epoch, pivot * t_max, a1)
    return cosine(epoch - pivot * t_max, (1 - pivot) * t_max, a2)


class Scheduler:
    """Updates optimizer's learning rates using provided scheduling function."""
    def __init__(self, opt, schedule):
        self.opt = opt
        self.schedule = schedule
        self.history = defaultdict(list)
        self.lr = None

    def step(self, t):
        for i, group in enumerate(self.opt.param_groups):
            self.lr = self.opt.defaults['lr'] * self.schedule(t)
            group['lr'] = self.lr
            self.history[i].append(self.lr)

    def get_last_lr(self):
        return self.lr
