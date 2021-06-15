"""Define generic classes and functions to facilitate baseline construction"""
import os
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset

__author__ = "Mario Morvan"
__email__ = "mario.morvan.18@ucl.ac.uk"

class ArielMLDataset(Dataset):
    """Class for reading files for the Ariel ML data challenge 2021"""

    def __init__(self, lc_path, params_path=None, transform=None, start_ind=0,
                 max_size=int(1e9), shuffle=True, seed=None, device=None):
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
        self.files = self.files[start_ind:start_ind+max_size]

        if params_path is not None:
            self.params_path = params_path
        else:
            self.params_path = None
            self.params_files = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item_lc_path = Path(self.lc_path) / self.files[idx]
        lc = torch.from_numpy(np.loadtxt(item_lc_path))
        if self.transform:
            lc = self.transform(lc)
        if self.params_path is not None:
            item_params_path = Path(self.params_path) / self.files[idx]
            target = torch.from_numpy(np.loadtxt(item_params_path))
        else:
            target = torch.Tensor()
        return {'lc': lc.to(self.device),
                'target': target.to(self.device)}


def simple_transform(x):
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

        return (1e4 - 2 * (weights * y * torch.abs(pred - y)).sum() / weights.sum() * 1e6)