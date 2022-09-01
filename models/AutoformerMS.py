# Copyright (c) 2019-present, Royal Bank of Canada.
# Copyright (c) 2021 THUML @ Tsinghua University
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the Autoformer (https://arxiv.org/pdf/2106.13008.pdf) implementation
# from https://github.com/thuml/Autoformer by THUML @ Tsinghua University
####################################################################################

import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_mine
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp

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
    Multi-scale version of Autoformer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        self.decomp = series_decomp(configs.moving_avg)

        # Embedding
        # We use our new DataEmbedding which incldues the scale information
        self.enc_embedding = DataEmbedding_mine(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_mine(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout, is_decoder=True)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg = configs.moving_avg,
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
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg = configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        """
        following functions will be used to manage scales
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
                if self.input_decomposition_type == 1:
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
