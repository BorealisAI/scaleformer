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
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import ReformerLayer
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
    Reformer with O(LlogL) complexity
    - It is notable that Reformer is not proposed for time series forecasting, in that it cannot accomplish the cross attention.
    - Here is only one adaption in BERT-style, other possible implementations can also be acceptable.
    - The hyper-parameters, such as bucket_size and n_hashes, need to be further tuned.
    The official repo of Reformer (https://github.com/lucidrains/reformer-pytorch) can be very helpful, if you have any questiones.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding_mine(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout, is_decoder=True)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, configs.d_model, configs.n_heads, bucket_size=configs.bucket_size,
                                  n_hashes=configs.n_hashes),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        """
        following functions will be used to manage scales
        """
        self.scale_factor = configs.scale_factor
        self.scales = configs.scales
        self.mv = moving_avg()
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='linear')

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
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
            enc_out = enc_out - mean
            dec_out = dec_out - mean

            # add placeholder
            enc_out = torch.cat([enc_out, dec_out[:, -self.pred_len//scale:, :]], dim=1)

            # Reformer: encoder only
            enc_out = self.enc_embedding(enc_out, x_mark_enc[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=x_enc.shape[1])
            enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
            enc_out = self.projection(enc_out)

            enc_out = enc_out + mean
            dec_out_coarse = enc_out[:, -x_dec.shape[1]//scale:, :]
            outputs.append(dec_out_coarse[:, -self.pred_len//scale:, :])
        return outputs

