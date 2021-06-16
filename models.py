import numpy as np
import torch
from torch.nn import Module, Sequential
from constants import n_wavelengths, n_timesteps
from torch import nn

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
    def __init__(self, num_mlp_layers = 3, emb_dim = 400, drop_ratio = 0, input_dim=n_wavelengths*n_timesteps, output_dim=n_wavelengths):
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
        x = torch.flatten(x, start_dim=1)
        x = self.input_fc(x)
        x = self.mlp(x)
        output = self.output_fc(x)
        return output 


# Implemented
class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)

class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.
    
    The separable convlution is a method to reduce number of the parameters 
    in the deep learning network for slight decrease in predictions quality.
    """
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad, groups=ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad, drop=None,
                 activ=lambda: nn.ReLU(inplace=True)):
    
        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.layers(x)

class Classifier(nn.Module):
    def __init__(self, raw_ni, fft_ni, no, drop=.5):
        super().__init__()
        
        self.raw = nn.Sequential(
            SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop),
            SepConv1d(    32,  64, 8, 4, 2, drop=drop),
            SepConv1d(    64, 128, 8, 4, 2, drop=drop),
            SepConv1d(   128, 256, 8, 4, 2),
            Flatten(),
            nn.Dropout(drop), nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))
        
        self.fft = nn.Sequential(
            SepConv1d(fft_ni,  32, 8, 2, 4, drop=drop),
            SepConv1d(    32,  64, 8, 2, 4, drop=drop),
            SepConv1d(    64, 128, 8, 4, 4, drop=drop),
            SepConv1d(   128, 128, 8, 4, 4, drop=drop),
            SepConv1d(   128, 256, 8, 2, 3),
            Flatten(),
            nn.Dropout(drop), nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, no))
        
    def forward(self, t_raw, t_fft):
        raw_out = self.raw(t_raw)
        fft_out = self.fft(t_fft)
        t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(t_in)
        return out


class UConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride=1, pad=0, drop=None, bn=True, dilation=0,
                 activ=lambda: nn.PReLU()):
    
        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [nn.Conv1d(ni, no, kernel, stride, pad, dilation)]
        if activ:
            layers.append(activ())
        if bn:
            layers.append(nn.BatchNorm1d(no))
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.layers(x)

class DilatedNet(nn.Module):
    def __init__(self, in_channel=55, hidden_size=2000, dilation=2):
        """
        """
        super(DilatedNet, self).__init__()
        self.dilation = dilation
        self.hidden_size = hidden_size
        # Input
        self.cnn = nn.Sequential(
            UConv1d(in_channel, 64, kernel=3, pad=2, dilation=dilation),
            UConv1d(64, 128, kernel=3, pad=2, dilation=dilation),
            UConv1d(128, 256, kernel=3, pad=2, dilation=dilation)
        )

        self.flatten = Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(256*n_timesteps, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//4),
            nn.BatchNorm1d(hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, in_channel),
        )

    def forward(self, x):
        """

        :param x: Pytorch Variable
        :return:
        """
        out = self.cnn(x)
        out = self.flatten(out)
        out = self.mlp(out)
        return out