#!/bin/bash
# 
# Copyright (c) 2019-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --mem=40G
#SBATCH --gres=gpu:1

source .env/bin/activate

python -u run.py \
  --data_path $inp_data \
  --model $model \
  --pred_len $pred_len \
  --loss $loss

echo "DONE"
