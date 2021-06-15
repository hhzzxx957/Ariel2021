import numpy as np
import torch
from torch.nn import Module, Sequential
from constants import n_wavelengths, n_timesteps

class Baseline(Module):
    """Baseline model for Ariel ML data challenge 2021"""

    def __init__(self, H1=1024, H2=256, input_dim=n_wavelengths*n_timesteps, output_dim=n_wavelengths):
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
        self.network = Sequential(torch.nn.Linear(input_dim, H1),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(H1, H2),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(H2, output_dim),
                                  )

    def __call__(self, x):
        """Predict rp/rs from input tensor light curve x"""
        out = torch.flatten(
            x, start_dim=1)  # Need to flatten out the input light curves for this type network
        out = self.network(out)
        return out                             

class MLP(torch.nn.Module):
    """ MLP model"""
    def __init__(self, num_mlp_layers = 3, emb_dim = 384, drop_ratio = 0, input_dim=n_wavelengths*n_timesteps, output_dim=n_wavelengths):
        super(MLP, self).__init__()
        self.num_mlp_layers = num_mlp_layers
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio 

        # mlp
        input_module_list = [
            torch.nn.Linear(input_dim, self.emb_dim),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = self.drop_ratio),
        ]

        self.input_fc = torch.nn.Sequential(*input_module_list)

        module_list = []
        for _ in range(self.num_mlp_layers - 1):
            module_list += [torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = self.drop_ratio)]

        # module_list = [torch.nn.Linear(input_dim, 1)]

        self.mlp = torch.nn.Sequential(
            *module_list
        )

        # relu is applied in the last layer to ensure positivity
        output_module_list = [torch.nn.Linear(self.emb_dim, output_dim)]
        self.output_fc = torch.nn.Sequential(*output_module_list)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.mlp(x)
        output = self.output_fc(x)
        return output 

