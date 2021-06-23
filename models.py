import numpy as np
import torch
from torch import nn
from torch.nn import Module, Sequential
import torch.nn.functional as F

from constants import n_timesteps, n_wavelengths


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


class MLP(torch.nn.Module):
    """ MLP model"""
    def __init__(self,
                 num_mlp_layers=3,
                 emb_dim=400,
                 drop_ratio=0,
                 input_dim=n_wavelengths * n_timesteps,
                 output_dim=n_wavelengths):
        super(MLP, self).__init__()
        self.num_mlp_layers = num_mlp_layers
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio

        # mlp
        input_module_list = [
            torch.nn.Linear(input_dim, self.emb_dim),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.drop_ratio),
        ]

        self.input_fc = torch.nn.Sequential(*input_module_list)

        module_list = []
        for _ in range(self.num_mlp_layers - 1):
            module_list += [
                torch.nn.Linear(self.emb_dim, self.emb_dim),
                torch.nn.BatchNorm1d(self.emb_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=self.drop_ratio)
            ]

        # module_list = [torch.nn.Linear(input_dim, 1)]

        self.mlp = torch.nn.Sequential(*module_list)

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
            return x.reshape(x.size(0), -1)
        return x.reshape(-1)


class UConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self,
                 ni,
                 no,
                 kernel,
                 stride=1,
                 pad=0,
                 drop=None,
                 bn=True,
                 pool=True,
                 dilation=0,
                 activ=lambda: nn.PReLU()):

        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [nn.Conv1d(ni, no, kernel, stride, pad, dilation)]  #[_SepConv1d(ni, no, kernel, stride, pad, dilation)] #  #
        if activ:
            layers.append(activ())
        if bn:
            layers.append(nn.BatchNorm1d(no))
        if drop is not None:
            layers.append(nn.Dropout(drop))
        if pool:
            layers.append(nn.AvgPool1d(kernel_size=2, stride=2))  # AvgPool1d
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SkipConv1d(nn.Module):
    """
    """
    def __init__(self,
                 ni,
                 kernel,
                 stride=1,
                 pad=0,
                 drop=None,
                 bn=True,
                 dilation=0,
                 activ=lambda: nn.PReLU()):

        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [nn.Conv1d(ni, ni, kernel, stride, pad, dilation)]
        layers2 = [nn.Conv1d(ni, ni, kernel, stride, pad, dilation)]
        if bn:
            layers.append(nn.BatchNorm1d(ni))
            layers2.append(nn.BatchNorm1d(ni))
        if activ:
            layers.append(activ())
        self.layers = nn.Sequential(*layers)
        self.layers2 = nn.Sequential(*layers2)
        self.activ = activ()

    def forward(self, x):
        identity = x
        out = self.layers(x)
        out = self.layers2(out)
        out = self.activ(out + identity)
        return out


class AttentionConv(nn.Module):
    def __init__(self, ni, no, kernel, stride=1, pad=1, n_head=2, d_k=300, bias=False):
        super(AttentionConv, self).__init__()
        self.cnn = nn.Conv1d(ni, no, kernel, stride, pad)
        self.attention_heads = nn.MultiheadAttention(d_k, num_heads=n_head)

    def forward(self, x):
        out = self.cnn(x)
        out, attn = self.attention_heads(out, out, out)

        return out


class DilatedNet(nn.Module):
    def __init__(self, in_channel=55, hidden_size=2048, dilation=2, add_feat=False, add_lstm=False, add_attn=False):
        """
        """
        super(DilatedNet, self).__init__()
        self.dilation = dilation
        self.hidden_size = hidden_size
        self.add_feat = add_feat
        self.add_lstm = add_lstm
        self.add_attn = add_attn
        # Input
        self.cnn = nn.Sequential(
            UConv1d(in_channel, 64, kernel=3, pad=2, dilation=dilation),
            UConv1d(64, 64, kernel=3, pad=2, dilation=dilation),
            
            UConv1d(64, 128, kernel=3, pad=2, dilation=dilation),
            UConv1d(128, 128, kernel=3, pad=2, dilation=dilation),
            
            UConv1d(128, 256, kernel=3, pad=2, dilation=dilation),
            UConv1d(256, 256, kernel=3, pad=2, dilation=dilation),
            )
        self.flatten = Flatten()

        if add_lstm:
            self.lstm_block = BlockLSTM(300, 2)
            self.flatten2 = Flatten()

        if add_attn:
            self.attn_block = AttentionConv(in_channel, 16, 3)
            self.flatten3 = Flatten()

        self.mlp = nn.Sequential(
            nn.Linear(256 * (n_timesteps // 2**6), hidden_size),  # 
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
        )

        self.embed1 = nn.Embedding(10, 16)
        self.embed2 = nn.Embedding(10, 16)

        self.mlp_feat = nn.Sequential(
            nn.Linear(6, 64),  # 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        mlpout_input_size = hidden_size // 4
        if add_feat:
            mlpout_input_size += 16 #+55
        if add_lstm:
            mlpout_input_size += 256*2
        if add_attn:
            mlpout_input_size += 16*300

        self.mlpout = nn.Sequential(
            nn.Linear(mlpout_input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, in_channel),
        )
        self.init_weights(nn.init.kaiming_normal_)
        
    def init_weights(self, init_fn):
        def init(m): 
            for child in m.children():
                if isinstance(child, nn.Conv1d):
                    init_fn(child.weights)
        init(self)

    def forward(self, x):
        """

        :param x: Pytorch Variable
        :return:
        """
        feat_lgb = None
        if self.add_feat:
            x, feat = x
            # feat_file = feat[:, 6+55:6+55+3]
            feat_lgb = feat[:, 6:6+55]
            feat = feat[:,:6]
        out = self.cnn(x)
        out = self.flatten(out)
        out = self.mlp(out)

        if self.add_feat:
            feat_out = self.mlp_feat(feat)
            out = torch.cat([out, feat_out], axis=1) #feat_lgb
        if self.add_lstm:
            lstm_out = self.lstm_block(x)
            lstm_out = self.flatten2(lstm_out)
            out = torch.cat([out, lstm_out], axis=1)
        if self.add_attn:
            attn_out = self.attn_block(x)
            attn_out = self.flatten3(attn_out)
            out = torch.cat([out, attn_out], axis=1)

        out = self.mlpout(out)
        return out

class BlockLSTM(nn.Module):
    def __init__(self,
                 time_steps,
                 num_variables,
                 lstm_hs=256,
                 dropout=0.2,):
        super().__init__()
        self.lstm = nn.LSTM(input_size=time_steps,
                            hidden_size=lstm_hs,
                            num_layers=num_variables)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # input is of the form (batch_size, num_variables, time_steps), e.g. (128, 1, 512)
        x = torch.transpose(x, 0, 1)
        # lstm layer is of the form (num_variables, batch_size, time_steps)
        output,(h_n,c_n) = self.lstm(x)
        # dropout layer input shape:
        h_n = self.dropout(h_n)
        # output shape is of the form ()
        h_n = torch.transpose(h_n, 0, 1)
        return h_n


class DilatedCNNLSTMNet(nn.Module):
    def __init__(self, in_channel=55, hidden_size=2000, dilation=2, add_feat=False):
        """
        """
        super(DilatedCNNLSTMNet, self).__init__()
        self.dilation = dilation
        self.hidden_size = hidden_size
        self.add_feat = add_feat
        # Input
        self.lstm_block = BlockLSTM(300, 2)
        self.flatten2 = Flatten()

        self.cnn = nn.Sequential(
            UConv1d(in_channel, 64, kernel=3, pad=2, dilation=dilation),
            UConv1d(64, 64, kernel=3, pad=2, dilation=dilation),
            UConv1d(64, 128, kernel=3, pad=2, dilation=dilation),
            UConv1d(128, 128, kernel=3, pad=2, dilation=dilation),
            UConv1d(128, 256, kernel=3, pad=2, dilation=dilation),
            UConv1d(256, 256, kernel=3, pad=2, dilation=dilation),
            )
        self.flatten = Flatten()

        self.mlp = nn.Sequential(
            nn.Linear(256 * (n_timesteps // 2**6), hidden_size),  # 
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(6, 64),  # 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        if add_feat:
            mlpout_input_size = hidden_size // 4 + 16 + 256*2
        else:
            mlpout_input_size = hidden_size // 4 + 256*2
        self.mlpout = nn.Sequential(
            nn.Linear(mlpout_input_size, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, in_channel),
        )
        self.init_weights(nn.init.kaiming_normal_)
        
    def init_weights(self, init_fn):
        def init(m): 
            for child in m.children():
                if isinstance(child, nn.Conv1d):
                    init_fn(child.weights)
        init(self)

    def forward(self, x):
        """

        :param x: Pytorch Variable
        :return:
        """
        if self.add_feat:
            x, feat = x
        lstm_out = self.lstm_block(x)
        lstm_out = self.flatten2(lstm_out)

        out = self.cnn(x)
        out = self.flatten(out)
        out = self.mlp(out)
        if self.add_feat:
            feat_out = self.mlp2(feat)
            out = torch.cat([out, lstm_out, feat_out], axis=1)
        out = self.mlpout(out)
        return out
