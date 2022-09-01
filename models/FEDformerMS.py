# Copyright (c) 2019-present, Royal Bank of Canada.
# Copyright (c) 2021 DAMO Academy @ Alibaba
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the FEDformer (https://arxiv.org/abs/2201.12740) implementation
# from https://github.com/MAZiqing/FEDformer by DAMO Academy @ Alibaba
####################################################################################

import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_mine
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi


class moving_avg(nn.Module):
    """
    Downsample series using an average pooling
    """
    def __init__(self):
        super(moving_avg, self).__init__()

    def forward(self, x, scale=1):
        if x is None:
            return None
        x = nn.functional.avg_pool1d(x.permute(0, 2, 1), scale, scale)
        x = x.permute(0, 2, 1)
        return x


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_mine(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_mine(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout, is_decoder=True)

        if configs.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=configs.modes,
                                                  ich=configs.d_model,
                                                  base=configs.base,
                                                  activation=configs.cross_activation)
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select)
            decoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len//2+self.pred_len,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
                                                      out_channels=configs.d_model,
                                                      seq_len_q=self.seq_len//2+self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=configs.modes,
                                                      mode_select_method=configs.mode_select)
        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len//2))
        dec_modes = int(min(configs.modes, (configs.seq_len//2+configs.pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        """
        following functions will be used to manage scales and inputs
        """
        self.scale_factor = configs.scale_factor
        self.scales = configs.scales
        self.mv = moving_avg()
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='linear')
        self.input_decomposition_type = 1

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        scales = self.scales
        label_len = x_dec.shape[1]-self.pred_len
        outputs = []
        for scale in scales:
            enc_out = self.mv(x_enc, scale)
            if scale == scales[0]: # initialize the input of decoder at first step
                if self.input_decomposition_type == 0:
                    mean = enc_out.mean(1).unsqueeze(1)
                    enc_out = enc_out - mean
                    tmp_mean = torch.mean(enc_out, dim=1).unsqueeze(1).repeat(1, self.pred_len//scale, 1)
                    zeros = torch.zeros([x_dec.shape[0], self.pred_len//scale, x_dec.shape[2]], device=x_enc.device)
                    seasonal_init, trend_init = self.decomp(enc_out)
                    trend_init = torch.cat([trend_init[:, -self.label_len//scale:, :], tmp_mean], dim=1)
                    seasonal_init = torch.cat([seasonal_init[:, -self.label_len//scale:, :], zeros], dim=1)
                    dec_out = self.mv(x_dec, scale) - mean
                else:
                    dec_out = self.mv(x_dec, scale)
                    mean = enc_out.mean(1).unsqueeze(1)
                    enc_out = enc_out - mean
                    dec_out[:, :label_len//scale, :] = dec_out[:, :label_len//scale, :] - mean
            else: # generation the input at each scale and cross normalization
                dec_out = self.upsample(dec_out_coarse.detach().permute(0,2,1)).permute(0,2,1)
                dec_out[:, :label_len//scale, :] = self.mv(x_dec[:, :label_len, :], scale)
                mean = torch.cat((enc_out, dec_out[:, label_len//scale:, :]), 1).mean(1).unsqueeze(1)
                enc_out = enc_out - mean
                dec_out = dec_out - mean

            # redefining the inputs to the decoder to be scale aware
            trend_init = torch.zeros_like(dec_out)
            seasonal_init = dec_out

            enc_out = self.enc_embedding(enc_out, x_mark_enc[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
            enc_out, attns = self.encoder(enc_out)
            dec_out = self.dec_embedding(seasonal_init, x_mark_dec[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
            seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend=trend_init)
            dec_out_coarse = seasonal_part + trend_part

            dec_out_coarse = dec_out_coarse + mean
            outputs.append(dec_out_coarse[:, -self.pred_len//scale:, :])

        return outputs
