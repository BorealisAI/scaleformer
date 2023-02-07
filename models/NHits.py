# Copyright (c) 2019-present, Royal Bank of Canada.
# Copyright (c) 2022 Nixtla
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the N-HiTS (https://arxiv.org/abs/2201.12886) implementation
# from https://github.com/Nixtla/neuralforecast by Nixtla
####################################################################################

import math
from functools import partial
from typing import Tuple
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class CausalConv1d(nn.Module):
    """
    Receives x input of dim [N,C,T], computes a unidimensional
    causal convolution.
    Parameters
    ----------
    in_channels: int
    out_channels: int
    activation: str
        https://discuss.pytorch.org/t/call-activation-function-from-string
    padding: int
    kernel_size: int
    dilation: int
    Returns:
    x: tesor
        torch tensor of dim [N,C,T]
        activation(conv1d(inputs, kernel) + bias)
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, dilation, activation, stride:int=1, with_weight_norm:bool=False):
        super(CausalConv1d, self).__init__()
        assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'

        self.conv       = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding,
                                    dilation=dilation)
        if with_weight_norm: self.conv = weight_norm(self.conv)

        self.chomp      = Chomp1d(padding)
        self.activation = getattr(nn, activation)()
        self.causalconv = nn.Sequential(self.conv, self.chomp, self.activation)

    def forward(self, x):
        return self.causalconv(x)

class _TemporalBlock2(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(_TemporalBlock2, self).__init__()
        self.causalconv1 = CausalConv1d(in_channels=n_inputs, out_channels=n_outputs,
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        dilation=dilation, activation='ReLU', with_weight_norm=True)

        self.causalconv2 = CausalConv1d(in_channels=n_outputs, out_channels=n_outputs,
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        dilation=dilation, activation='ReLU', with_weight_norm=True)

        self.net = nn.Sequential(self.causalconv1, nn.Dropout(dropout),
                                 self.causalconv2, nn.Dropout(dropout))

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.causalconv1.conv.weight.data.normal_(0, 0.01)
        self.causalconv2.conv.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(_TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [_TemporalBlock2(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                       padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class RepeatVector(nn.Module):
    """
    Receives x input of dim [N,C], and repeats the vector
    to create tensor of shape [N, C, K]
    : repeats: int, the number of repetitions for the vector.
    """
    def __init__(self, repeats):
        super(RepeatVector, self).__init__()
        self.repeats = repeats

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1, 1, self.repeats) # <------------ Mejorar?
        return x

class Chomp1d(nn.Module):
    """
    Receives x input of dim [N,C,T], and trims it so that only
    'time available' information is used. Used for one dimensional
    causal convolutions.
    : param chomp_size: length of outsample values to skip.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class _StaticFeaturesEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(_StaticFeaturesEncoder, self).__init__()
        layers = [nn.Dropout(p=0.5),
                  nn.Linear(in_features=in_features, out_features=out_features),
                  nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x

class _sEncoder(nn.Module):
    def __init__(self, in_features, out_features, n_time_in):
        super(_sEncoder, self).__init__()
        layers = [nn.Dropout(p=0.5),
                  nn.Linear(in_features=in_features, out_features=out_features),
                  nn.ReLU()]
        self.encoder = nn.Sequential(*layers)
        self.repeat = RepeatVector(repeats=n_time_in)

    def forward(self, x):
        # Encode and repeat values to match time
        x = self.encoder(x)
        x = self.repeat(x) # [N,S_out] -> [N,S_out,T]
        return x

# Cell
class _IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, interpolation_mode: str):
        super().__init__()
        assert (interpolation_mode in ['linear','nearest']) or ('cubic' in interpolation_mode)
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:

        backcast = theta[:, :self.backcast_size]
        knots = theta[:, self.backcast_size:]

        if self.interpolation_mode=='nearest':
            knots = knots[:,None,:]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode)
            forecast = forecast[:,0,:]
        elif self.interpolation_mode=='linear':
            knots = knots[:,None,:]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode)
            forecast = forecast[:,0,:]
        elif 'cubic' in self.interpolation_mode:
            batch_size = len(backcast)
            knots = knots[:,None,None,:]
            forecast = t.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots)/batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(knots[i*batch_size:(i+1)*batch_size], size=self.forecast_size, mode='bicubic')
                forecast[i*batch_size:(i+1)*batch_size] += forecast_i[:,0,0,:]

        return backcast, forecast

# Cell
class _ExogenousBasisInterpretable(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast_basis = insample_x_t
        forecast_basis = outsample_x_t

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast

class _ExogenousBasisWavenet(nn.Module):
    def __init__(self, out_features, in_features, num_levels=4, kernel_size=3, dropout_prob=0):
        super().__init__()
        # Shape of (1, in_features, 1) to broadcast over b and t
        self.weight = nn.Parameter(t.Tensor(1, in_features, 1), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(0.5))

        padding = (kernel_size - 1) * (2**0)
        input_layer = [nn.Conv1d(in_channels=in_features, out_channels=out_features,
                                 kernel_size=kernel_size, padding=padding, dilation=2**0),
                                 Chomp1d(padding),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_prob)]
        conv_layers = []
        for i in range(1, num_levels):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation
            conv_layers.append(nn.Conv1d(in_channels=out_features, out_channels=out_features,
                                         padding=padding, kernel_size=3, dilation=dilation))
            conv_layers.append(Chomp1d(padding))
            conv_layers.append(nn.ReLU())
        conv_layers = input_layer + conv_layers

        self.wavenet = nn.Sequential(*conv_layers)

    def transform(self, insample_x_t, outsample_x_t):
        n_time_in = insample_x_t.shape[2]

        x_t = t.cat([insample_x_t, outsample_x_t], dim=2)

        x_t = x_t * self.weight # Element-wise multiplication, broadcasted on b and t. Weights used in L1 regularization
        x_t = self.wavenet(x_t)[:]

        backcast_basis = x_t[:,:, :n_time_in]
        forecast_basis = x_t[:,:, n_time_in:]

        return backcast_basis, forecast_basis

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast

class _ExogenousBasisTCN(nn.Module):
    def __init__(self, out_features, in_features, num_levels = 4, kernel_size=2, dropout_prob=0):
        super().__init__()
        n_channels = num_levels * [out_features]
        self.tcn = _TemporalConvNet(num_inputs=in_features, num_channels=n_channels, kernel_size=kernel_size, dropout=dropout_prob)

    def transform(self, insample_x_t, outsample_x_t):
        n_time_in = insample_x_t.shape[2]

        x_t = t.cat([insample_x_t, outsample_x_t], dim=2)

        x_t = self.tcn(x_t)[:]
        backcast_basis = x_t[:,:, :n_time_in]
        forecast_basis = x_t[:,:, n_time_in:]

        return backcast_basis, forecast_basis

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast

# Cell
def _init_weights(module, initialization):
    if type(module) == t.nn.Linear:
        if initialization == 'orthogonal':
            t.nn.init.orthogonal_(module.weight)
        elif initialization == 'he_uniform':
            t.nn.init.kaiming_uniform_(module.weight)
        elif initialization == 'he_normal':
            t.nn.init.kaiming_normal_(module.weight)
        elif initialization == 'glorot_uniform':
            t.nn.init.xavier_uniform_(module.weight)
        elif initialization == 'glorot_normal':
            t.nn.init.xavier_normal_(module.weight)
        elif initialization == 'lecun_normal':
            pass #t.nn.init.normal_(module.weight, 0.0, std=1/np.sqrt(module.weight.numel()))
        else:
            assert 1<0, f'Initialization {initialization} not found'

# Cell
ACTIVATIONS = ['ReLU',
               'Softplus',
               'Tanh',
               'SELU',
               'LeakyReLU',
               'PReLU',
               'Sigmoid']

class _NHITSBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self, n_time_in: int, n_time_out: int, n_x: int,
                 n_s: int, n_s_hidden: int, n_theta: int, n_mlp_units: list,
                 n_pool_kernel_size: int, pooling_mode: str, basis: nn.Module,
                 n_layers: int,  batch_normalization: bool, dropout_prob: float, activation: str):
        """
        """
        super().__init__()

        assert (pooling_mode in ['max','average'])

        n_time_in_pooled = int(np.ceil(n_time_in/n_pool_kernel_size))

        if n_s == 0:
            n_s_hidden = 0
        n_mlp_units = [n_time_in_pooled + (n_time_in+n_time_out)*n_x + n_s_hidden] + n_mlp_units

        self.n_time_in = n_time_in
        self.n_time_out = n_time_out
        self.n_s = n_s
        self.n_s_hidden = n_s_hidden
        self.n_x = n_x
        self.n_pool_kernel_size = n_pool_kernel_size
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob

        assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'
        activ = getattr(nn, activation)()

        if pooling_mode == 'max':
            self.pooling_layer = nn.MaxPool1d(kernel_size=self.n_pool_kernel_size,
                                              stride=self.n_pool_kernel_size, ceil_mode=True)
        elif pooling_mode == 'average':
            self.pooling_layer = nn.AvgPool1d(kernel_size=self.n_pool_kernel_size,
                                              stride=self.n_pool_kernel_size, ceil_mode=True)

        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(nn.Linear(in_features=n_mlp_units[i], out_features=n_mlp_units[i+1]))
            hidden_layers.append(activ)

            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=n_mlp_units[i+1]))

            if self.dropout_prob>0:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=n_mlp_units[-1], out_features=n_theta)]
        layers = hidden_layers + output_layer

        # n_s is computed with data, n_s_hidden is provided by user, if 0 no statics are used
        if (self.n_s > 0) and (self.n_s_hidden > 0):
            self.static_encoder = _StaticFeaturesEncoder(in_features=n_s, out_features=n_s_hidden)
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: t.Tensor, insample_x_t: t.Tensor,
                outsample_x_t: t.Tensor, x_s: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:

        insample_y = insample_y.unsqueeze(1)
        insample_y = self.pooling_layer(insample_y)
        insample_y = insample_y.squeeze(1)

        batch_size = len(insample_y)
        if self.n_x > 0:
            insample_y = t.cat(( insample_y, insample_x_t.reshape(batch_size, -1) ), 1)
            insample_y = t.cat(( insample_y, outsample_x_t.reshape(batch_size, -1) ), 1)

        # Static exogenous
        if (self.n_s > 0) and (self.n_s_hidden > 0):
            x_s = self.static_encoder(x_s)
            insample_y = t.cat((insample_y, x_s), 1)

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta, insample_x_t, outsample_x_t)

        return backcast, forecast

# Cell
class Model(nn.Module):
    """
    NHITS Model.
    """
    def __init__(self,
                 configs, 
                 n_s=0,
                 n_x=0,
                 n_s_hidden=0,
                 n_x_hidden=0,
                 stack_types=('identity', 'identity', 'identity'),
                 n_blocks=(1, 1, 1),
                 n_layers=(2, 2, 2, 2, 2, 2, 2, 2, 2),
                 n_mlp_units=[[512, 512], [512, 512], [512, 512]],
                 n_pool_kernel_size=(1, 1, 1),
                 n_freq_downsample=(24, 12, 1),
                 pooling_mode='max',
                 interpolation_mode='linear',
                 dropout_prob_theta=0.,
                 activation='ReLU',
                 initialization='lecun_normal',
                 batch_normalization=True,
                 shared_weights=False):
        super().__init__()

        self.n_time_in = configs.seq_len
        self.label_len = configs.label_len
        self.n_time_out = configs.pred_len

        blocks = self.create_stack(stack_types=stack_types,
                                   n_blocks=n_blocks,
                                   n_time_in=self.n_time_in,
                                   n_time_out=self.n_time_out,
                                   n_x=n_x,
                                   n_x_hidden=n_x_hidden,
                                   n_s=n_s,
                                   n_s_hidden=n_s_hidden,
                                   n_layers=n_layers,
                                   n_mlp_units=n_mlp_units,
                                   n_pool_kernel_size=n_pool_kernel_size,
                                   n_freq_downsample=n_freq_downsample,
                                   pooling_mode=pooling_mode,
                                   interpolation_mode=interpolation_mode,
                                   batch_normalization=batch_normalization,
                                   dropout_prob_theta=dropout_prob_theta,
                                   activation=activation,
                                   shared_weights=shared_weights,
                                   initialization=initialization)
        self.blocks = t.nn.ModuleList(blocks)

    def create_stack(self, stack_types, n_blocks,
                     n_time_in, n_time_out,
                     n_x, n_x_hidden, n_s, n_s_hidden,
                     n_layers, n_mlp_units,
                     n_pool_kernel_size, n_freq_downsample, pooling_mode, interpolation_mode,
                     batch_normalization, dropout_prob_theta,
                     activation, shared_weights, initialization):
        block_list = []
        for i in range(len(stack_types)):
            assert stack_types[i] in ['identity', 'exogenous', 'exogenous_tcn', 'exogenous_wavenet'], 'f Invalid stack type {stack_types[i]}'
            for block_id in range(n_blocks[i]):

                # Batch norm only on first block
                if (len(block_list)==0) and (batch_normalization):
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False

                # Shared weights
                if shared_weights and block_id>0:
                    nbeats_block = block_list[-1]
                else:
                    if stack_types[i] == 'identity':
                        n_theta = (self.n_time_in + max(self.n_time_out//n_freq_downsample[i], 1) )
                        basis = _IdentityBasis(backcast_size=n_time_in,
                                              forecast_size=n_time_out,
                                              interpolation_mode=interpolation_mode)

                    elif stack_types[i] == 'exogenous':
                        n_theta = 2 * n_x
                        basis = _ExogenousBasisInterpretable()

                    elif stack_types[i] == 'exogenous_tcn':
                        n_theta = 2 * n_x_hidden
                        basis = _ExogenousBasisTCN(n_x_hidden, n_x)

                    elif stack_types[i] == 'exogenous_wavenet':
                        n_theta = 2 * n_x_hidden
                        basis = _ExogenousBasisWavenet(n_x_hidden, n_x)

                    nbeats_block = _NHITSBlock(n_time_in=self.n_time_in,
                                               n_time_out=self.n_time_out,
                                               n_x=n_x,
                                               n_s=n_s,
                                               n_s_hidden=n_s_hidden,
                                               n_theta=n_theta,
                                               n_mlp_units=n_mlp_units[i],
                                               n_pool_kernel_size=n_pool_kernel_size[i],
                                               pooling_mode=pooling_mode,
                                               basis=basis,
                                               n_layers=n_layers[i],
                                               batch_normalization=batch_normalization_block,
                                               dropout_prob=dropout_prob_theta,
                                               activation=activation)

                # Select type of evaluation and apply it to all layers of block
                init_function = partial(_init_weights, initialization=initialization)
                nbeats_block.layers.apply(init_function)
                block_list.append(nbeats_block)
        return block_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_decomposition=False, **kwargs):

        # insample
        insample_y = x_enc.permute([0, 2, 1]).reshape([-1, self.n_time_in])
        insample_x_t = x_mark_enc.permute([0, 2, 1]).reshape([-1, self.n_time_in])
        insample_mask = t.ones_like(insample_y)


        # outsample
        outsample_y = x_dec[:, -self.n_time_out:, :].permute([0, 2, 1]).reshape([-1, self.n_time_out])
        outsample_x_t = x_mark_dec[:, -self.n_time_out:, :].permute([0, 2, 1]).reshape([-1, self.n_time_out])
        outsample_mask = t.ones_like(outsample_y)
        S = None

        if return_decomposition:
            forecast, block_forecasts = self.forecast_decomposition(insample_y=insample_y,
                                                                    insample_x_t=insample_x_t,
                                                                    insample_mask=insample_mask,
                                                                    outsample_x_t=outsample_x_t,
                                                                    x_s=S)
            return outsample_y, forecast, block_forecasts, outsample_mask

        else:
            forecast = self.forecast(insample_y=insample_y,
                                     insample_x_t=insample_x_t,
                                     insample_mask=insample_mask,
                                     outsample_x_t=outsample_x_t,
                                     x_s=S)
            forecast = forecast.view([x_dec.shape[0], -1, self.n_time_out]).permute([0, 2, 1])
            return forecast

    def forecast(self, insample_y: t.Tensor, insample_x_t: t.Tensor, insample_mask: t.Tensor,
                 outsample_x_t: t.Tensor, x_s: t.Tensor):

        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:] # Level with Naive1
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals, insample_x_t=insample_x_t,
                                             outsample_x_t=outsample_x_t, x_s=x_s)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast

        return forecast

    def forecast_decomposition(self, insample_y: t.Tensor, insample_x_t: t.Tensor, insample_mask: t.Tensor,
                               outsample_x_t: t.Tensor, x_s: t.Tensor):

        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        n_batch, n_channels, n_t = outsample_x_t.size(0), outsample_x_t.size(1), outsample_x_t.size(2)

        level = insample_y[:, -1:] # Level with Naive1
        block_forecasts = [ level.repeat(1, n_t) ]

        forecast = level
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals, insample_x_t=insample_x_t,
                                             outsample_x_t=outsample_x_t, x_s=x_s)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)

        # (n_batch, n_blocks, n_t)
        block_forecasts = t.stack(block_forecasts)
        block_forecasts = block_forecasts.permute(1,0,2)

        return forecast, block_forecasts
