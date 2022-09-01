#!/usr/bin/bash
# 
# Copyright (c) 2019-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# running for 4 datasets weather, traffic, electricity, exchange_rate
PRED_LENS=(96 192 336 720)
LOSS=(adaptive)
MODELS=(AutoformerMS InformerMS ReformerMS FEDformerMS PerformerMS)
DATASETS=(weather.csv traffic.csv electricity.csv exchange_rate.csv)
for loss in ${LOSS[@]};do
    for pred_len in ${PRED_LENS[@]} ; do
        for model in ${MODELS[@]} ; do
            for inp_data in ${DATASETS[@]}; do
                export model
                export inp_data
                export pred_len
                export seed
                export loss
                sbatch -o slurm/${inp_data}_${model}_${pred_len}_${seed}_${loss}.out run_single.sh
            done
        done
    done
done

PRED_LENS=(96 192 336 720)
LOSS=(mse)
MODELS=(Autoformer Informer Reformer FEDformer Performer)
DATASETS=(weather.csv traffic.csv electricity.csv exchange_rate.csv)
for loss in ${LOSS[@]};do
    for pred_len in ${PRED_LENS[@]} ; do
        for model in ${MODELS[@]} ; do
            for inp_data in ${DATASETS[@]}; do
                export model
                export inp_data
                export pred_len
                export seed
                export loss
                sbatch -o slurm/${inp_data}_${model}_${pred_len}_${seed}_${loss}.out run_single.sh
            done
        done
    done
done

# running for ILI dataset with new pred lens
PRED_LENS=(24 32 48 64)
LOSS=(adaptive)
MODELS=(AutoformerMS InformerMS ReformerMS FEDformerMS PerformerMS)
DATASETS=(national_illness.csv)
for loss in ${LOSS[@]};do
    for pred_len in ${PRED_LENS[@]} ; do
        for model in ${MODELS[@]} ; do
            for inp_data in ${DATASETS[@]}; do
                export model
                export inp_data
                export pred_len
                export seed
                export loss
                sbatch -o slurm/${inp_data}_${model}_${pred_len}_${seed}_${loss}.out run_single.sh
            done
        done
    done
done

PRED_LENS=(24 32 48 64)
LOSS=(mse)
MODELS=(Autoformer Informer Reformer FEDformer Performer)
DATASETS=(national_illness.csv)
for loss in ${LOSS[@]};do
    for pred_len in ${PRED_LENS[@]} ; do
        for model in ${MODELS[@]} ; do
            for inp_data in ${DATASETS[@]}; do
                export model
                export inp_data
                export pred_len
                export seed
                export loss
                sbatch -o slurm/${inp_data}_${model}_${pred_len}_${seed}_${loss}.out run_single.sh
            done
        done
    done
done
