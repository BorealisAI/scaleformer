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
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding_mine

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
    Multi-scale version of Informer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.prob_forecasting = configs.prob_forecasting
        c_out = configs.c_out*2 if self.prob_forecasting else configs.c_out

        # Embedding
        # We use our new DataEmbedding which incldues the scale information
        self.enc_embedding = DataEmbedding_mine(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_mine(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout, is_decoder=True)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, c_out, bias=True)
        )
        """
        following functions will be used to manage scales
        """
        self.scale_factor = configs.scale_factor
        self.scales = configs.scales
        self.mv = moving_avg()
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='linear')
        self.use_stdev_norm = False

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        label_len = x_dec.shape[1]-self.pred_len
        scales = self.scales
        outputs = []
        for scale in scales:
            enc_out = self.mv(x_enc, scale)
            if scale == scales[0]: # initialization
                dec_out = self.mv(x_dec, scale)
            else: # upsampling of the output from the previous steps
                dec_out = self.upsample(dec_out_coarse.detach().permute(0,2,1)).permute(0,2,1)
                dec_out[:, :label_len//scale, :] = self.mv(x_dec[:, :label_len, :], scale)

            # cross-scale normalization
            mean = torch.cat((enc_out, dec_out[:, label_len//scale:, :]), 1).mean(1).unsqueeze(1)
            if self.use_stdev_norm:
                stdev = torch.sqrt(torch.var(torch.cat((enc_out, dec_out[:, label_len//scale:, :]), 1), dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
                enc_out = enc_out / stdev
                dec_out = dec_out / stdev
            enc_out = enc_out - mean
            dec_out = dec_out - mean

            enc_out = self.enc_embedding(enc_out, x_mark_enc[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
            enc_out, attns = self.encoder(enc_out)
            dec_out = self.dec_embedding(dec_out, x_mark_dec[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
            dec_out_coarse = self.decoder(dec_out, enc_out, x_mask=self.mv(dec_self_mask, scale), cross_mask=self.mv(dec_enc_mask, scale))

            if self.prob_forecasting:
                out_scale = torch.nn.functional.softplus(dec_out_coarse[:,:,dec_out_coarse.shape[2]//2:])
                dec_out_coarse = dec_out_coarse[:,:,:dec_out_coarse.shape[2]//2]
                if self.use_stdev_norm:
                    dec_out_coarse = dec_out_coarse * stdev + mean
                else:
                    dec_out_coarse = dec_out_coarse + mean
                outputs.append(torch.cat((dec_out_coarse[:, -self.pred_len//scale:, :], out_scale[:, -self.pred_len//scale:, :]), 2))
            else:
                if self.use_stdev_norm:
                    dec_out_coarse = dec_out_coarse * stdev + mean
                else:
                    dec_out_coarse = dec_out_coarse + mean
                outputs.append(dec_out_coarse[:, -self.pred_len//scale:, :])
        return outputs
