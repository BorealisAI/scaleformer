

# Scaleformer
**Scaleformer: Iterative Multi-scale Refining Transformers for Time Series Forecasting**

This repo is a modified version of the public implementation of [Autoformer paper](https://arxiv.org/abs/2106.13008) which can be find in this [repository](https://github.com/thuml/Autoformer). We also use the related parts of FEDformer implementation from its [repository](https://github.com/MAZiqing/FEDformer).

## Why Scaleformer?
Using iteratively refining a forecasted time series at multiple scales with shared weights, architec- ture adaptations and a specially-designed normalization scheme, we are able to achieve significant performance improvements with minimal additional computational overhead.

<p align="center">
<img src="figs\teaser.png" width=90% alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of the proposed framework. (<b>Left</b>) Representation of a single scaling block. In each step, we pass the normalized upsampled version of the output from previous step along with the normalized downsampled version of encoder as the input. (<b>Right</b>) Representation of the full architecture. We process the input in a multi-scale manner iteratively from the smallest scale to the original scale.
</p>


Our experiments on four public datasets show that the proposed multi-scale framework
outperforms the corresponding baselines with an average improvement of 13% and
38% over Autoformer and Informer, respectively.

<p align="center">
<img src="figs\table.png" width=90% alt="" align=center />
<br><br>
<b>Table 1.</b> Comparison of the MSE and MAE results for our proposed multi-scale framework version of Informer and Autoformer (<b>-MSA</b>) with their original models as the baseline. Results are given in the multi-variate setting, for different lenghts of the horizon window. The look-back window size is fixed to 96 for all experiments. The best results are shown in <b>Bold</b>. Our method outperforms vanilla versions of both Informer and Autoformer over almost all datasets and settings.
</p>

## Installation
**1. Clone our repo and install the requirements:**
```
git clone https://github.com/BorealisAI/scaleformer.git
cd scaleformer
pip install -r requirements.txt
```
**2. Download datasets and create the dataset directory**
You can download the datasets from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/) or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing) links provided by [Autoformer](https://github.com/thuml/Autoformer) repository. For more information, please visit the repository.

Put all of the downloaded datasets in a `dataset` folder in the current directory:

```
scaleformer
├── dataset
│   ├── exchange_rate
|   |   └── exchange_rate.csv
│   ├── traffic
|   |   └── traffic.csv
|   └── ...
└── data_provider
└── exp
└── ...
```

## Running the code

**1. running a single experiment**

You can run a single experiment using the following command:
```
python -u run.py --data_path {DATASET} --model {MODEL} --pred_len {L} --loss {LOSS_FUNC}
```
for example, for using **Informer-MSA** as the model for **traffic** dataset with an output length of **192** and **adaptive** loss function, you can run:
```
python -u run.py --data_path traffic.csv --model InformerMS --pred_len 192 --loss adaptive
```
To see more examples and parameters, please see `run_all.sh`. 

**2. Running all of the experiments**

To run all of the experiments using slurm, you can use `run_all.sh` which uses `run_single.sh` to submit jobs with different parameters. The final errors of experiments will be available in `results.txt` and you can check `slurm` directory for the log of each experiment.


## Contact

If you have any question regarding the ScaleFormer, please contact aminshabaany@gmail.com.

## Citation

```
@article{shabani2022scaleformer,
  title={Scaleformer: Iterative Multi-scale Refining Transformers for Time Series Forecasting},
  author={Shabani, Amin and Abdi, Amir and Meng, Lili and Sylvain, Tristan},
  journal={arXiv preprint arXiv:2206.04038},
  year={2022}
}
```

## Acknowledgement

We acknowledge the following github repositories that made the base of our work:

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/jonbarron/robust_loss_pytorch.git
