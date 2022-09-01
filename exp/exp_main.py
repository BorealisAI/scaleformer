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

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, AutoformerMS, InformerMS, Reformer, \
        ReformerMS, FEDformer, FEDformerMS, Performer, PerformerMS
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from robust_loss_pytorch import AdaptiveLossFunction

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import math
import warnings

warnings.filterwarnings('ignore')



class CRPSMetric:
    """
    Compute the Continuous Ranked Probability Score (CRPS) is one of the most widely used error metrics to evaluate
    the quality of probabilistic regression tasks.
    Original paper: Gneiting, T. and Raftery, A.E., 2007. Strictly proper scoring rules, prediction, and estimation.
    Journal of the American statistical Association, 102(477), pp.359-378.
    https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
    Many distributions have closed form solution:
    https://mran.microsoft.com/snapshot/2017-12-13/web/packages/scoringRules/vignettes/crpsformulas.html
    """

    def __init__(self, x, loc, scale):
        self.value = x
        self.loc = loc
        self.scale = scale

    def gaussian_pdf(self, x):
        """Probability density function of a univariate standard Gaussian
            distribution with zero mean and unit variance.
        """
        _normconst = 1.0 / math.sqrt(2.0 * math.pi)
        return _normconst * torch.exp(-(x * x) / 2.0)

    def gaussian_cdf(self, x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def laplace_crps(self):
        """
        Compute the CRPS of observations x relative to laplace distributed forecasts with mean and b.
        Formula taken from Equation 2.1 Laplace distribution
        https://mran.microsoft.com/snapshot/2017-12-13/web/packages/scoringRules/vignettes/crpsformulas.html
        Returns:
        ----------
        crps: torch tensor
        The CRPS of each observation x relative to loc and scale.
        """
        # standadized value
        sx = (self.value - self.loc) / self.scale
        crps = self.scale * (sx.abs() + torch.exp(-sx.abs()) - 0.75)
        return crps

    def gaussian_crps(self):
        """
        Compute the CRPS of observations x relative to gaussian distributed forecasts with mean, sigma.
        CRPS(N(mu, sig^2); x)
        Formula taken from Equation (5):
        Calibrated Probablistic Forecasting Using Ensemble Model Output
        Statistics and Minimum CRPS Estimation. Gneiting, Raftery,
        Westveld, Goldman. Monthly Weather Review 2004
        http://journals.ametsoc.org/doi/pdf/10.1175/MWR2904.1
        Returns:
        ----------
        crps:
        The CRPS of each observation x relative to loc and scale.
        """
        # standadized value
        sx = (self.value - self.loc) / self.scale
        pdf = self.gaussian_pdf(sx)
        cdf = self.gaussian_cdf(sx)
        pi_inv = 1.0 / math.sqrt(math.pi)
        # the actual crps
        crps = self.scale * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
        return crps

def prob_loss_fn(mu, sigma, labels):
    '''
    from: https://github.com/husnejahan/DeepAR-pytorch/blob/master/net.py
    Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
    Args:
        mu: (Variable) dimension [batch_size] - estimated mean at time step t
        sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
        labels: (Variable) dimension [batch_size] z_t
    Returns:
        loss: (Variable) average log-likelihood loss across the batch
    '''
    zero_index = (labels != 0)
    distribution = torch.distributions.normal.Normal(mu[zero_index], sigma[zero_index])
    likelihood = distribution.log_prob(labels[zero_index])
    return -torch.mean(likelihood)

class moving_avg(nn.Module):
    def __init__(self):
        super(moving_avg, self).__init__()
    def forward(self, x, kernel_size):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            convert_numpy = True
            x = torch.tensor(x)
        else:
            convert_numpy = False
        x = nn.functional.avg_pool1d(x.permute(0, 2, 1), kernel_size, kernel_size)
        x = x.permute(0, 2, 1)
        if convert_numpy:
            x = x.numpy()
        return x

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.mv = moving_avg()

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'AutoformerMS': AutoformerMS,
            'Transformer': Transformer,
            'Informer': Informer,
            'InformerMS': InformerMS,
            'Reformer': Reformer,
            'ReformerMS': ReformerMS,
            'FEDformer': FEDformer,
            'FEDformerMS': FEDformerMS,
            'Performer': Performer,
            'PerformerMS': PerformerMS,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        print(f"NUMBER OF PARAMETERS IN MODEL: {self.args.model}: {sum(p.numel() for p in model.parameters())}")
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, additional_params=None):
        if additional_params is not None:
            model_optim = optim.Adam(list(self.model.parameters())+additional_params, lr=self.args.learning_rate)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs_all = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.prob_forecasting:
                            if self.args.use_multi_scale:
                                outputs_scale_all = [x[:, :, x.shape[2]//2:] for x in outputs_all]
                                outputs_scale = outputs_scale_all[-1]
                                outputs_all = [x[:, :, :x.shape[2]//2] for x in outputs_all]
                            else:
                                outputs_scale = outputs_all[:, :, outputs_all.shape[2]//2:]
                                outputs_all = outputs_all[:, :, :outputs_all.shape[2]//2]
                        if self.args.use_multi_scale:
                            outputs = outputs_all[-1]
                        else:
                            outputs = outputs_all
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                assert pred.shape==true.shape
                if self.args.prob_forecasting:
                    outputs_scale = outputs_scale.detach().cpu()
                    loss = criterion(pred, outputs_scale, true)
                else:
                    loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()


        criterion = self._select_criterion()
        if self.args.loss=='mse':
            criterion_tmp = torch.nn.MSELoss(reduction='none')
        elif self.args.loss=='huber':
            criterion_tmp = torch.nn.HuberLoss(reduction='none', delta=0.5)
        elif self.args.loss=='l1':
            criterion_tmp = torch.nn.L1Loss(reduction='none')
        elif self.args.loss=='adaptive':
            adaptive = AdaptiveLossFunction(1, torch.float32, self.device, alpha_hi=3.0)
            criterion_tmp = adaptive.lossfun 
            adaptive_optim = optim.Adam(list(adaptive.parameters()), lr=0.001)

        if self.args.prob_forecasting:
            criterion = prob_loss_fn
            criterion_tmp = prob_loss_fn

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            if self.args.loss=='adaptive':
                adaptive.print()
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                if self.args.loss=='adaptive':
                    adaptive_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs_all = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                        if self.args.prob_forecasting:
                            if self.args.use_multi_scale:
                                outputs_scale_all = [x[:, :, x.shape[2]//2:] for x in outputs_all]
                                outputs_scale = outputs_scale_all[-1]
                                outputs_all = [x[:, :, :x.shape[2]//2] for x in outputs_all]
                            else:
                                outputs_scale = outputs_all[:, :, outputs_all.shape[2]//2:]
                                outputs_all = outputs_all[:, :, :outputs_all.shape[2]//2]
                        if self.args.use_multi_scale:
                            outputs = outputs_all[-1]
                        else:
                            outputs = outputs_all


                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    assert outputs.shape==batch_y.shape, f"{outputs.shape}, {batch_y.shape}"

                    if self.args.loss=='adaptive':
                        loss =  criterion_tmp((outputs-batch_y).flatten().unsqueeze(-1))
                    else:
                        if self.args.prob_forecasting:
                            loss =  criterion_tmp(outputs, outputs_scale, batch_y)
                        else:
                            loss =  criterion_tmp(outputs, batch_y)
                    loss = loss.mean()
                    train_loss.append(loss.item())

                    if self.args.use_multi_scale:
                        for li, (scale, output) in enumerate(zip(self.args.scales[:-1], outputs_all[:-1])):
                            tmp_y = self.mv(batch_y, scale)
                            if self.args.loss=='adaptive':
                                tmp_loss = criterion_tmp((output - tmp_y).flatten().unsqueeze(-1))
                            else:
                                if self.args.prob_forecasting:
                                    tmp_loss = criterion_tmp(output, outputs_scale_all[li], tmp_y)
                                else:
                                    tmp_loss = criterion_tmp(output, tmp_y)
                            loss += tmp_loss.mean()/scale
                        loss = loss/2

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    if self.args.loss=='adaptive':
                        adaptive_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        if self.args.prob_forecasting:
            crps_total = []
            NLL_total = []

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        running_times = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                start_time = time.time()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs_all = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.prob_forecasting:
                            if self.args.use_multi_scale:
                                outputs_scale_all = [x[:, :, x.shape[2]//2:] for x in outputs_all]
                                outputs_scale = outputs_scale_all[-1]
                                outputs_all = [x[:, :, :x.shape[2]//2] for x in outputs_all]
                            else:
                                outputs_scale = outputs_all[:, :, outputs_all.shape[2]//2:]
                                outputs_all = outputs_all[:, :, :outputs_all.shape[2]//2]
                        if self.args.use_multi_scale:
                            outputs = outputs_all[-1]
                        else:
                            outputs = outputs_all
                running_times.append(time.time()-start_time)

                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()

                pred = outputs.numpy()
                true = batch_y.numpy()

                preds.append(pred)
                trues.append(true)
                if self.args.prob_forecasting:
                    outputs_scale = outputs_scale.detach().cpu()
                    metric_crps = CRPSMetric(batch_y, outputs, outputs_scale)
                    gaussian_crps = metric_crps.gaussian_crps()
                    crps_total.append(gaussian_crps.numpy())
                    NLL_total.append(prob_loss_fn(outputs, outputs_scale, batch_y))

                save_every = len(test_loader)//20 if len(test_loader)>20 else 1
                if i % save_every == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.svg'))

                    if self.args.use_multi_scale:
                        for scale, output in zip(self.args.scales, outputs_all):
                            gt = np.concatenate((self.mv(input, scale)[0, :, -1], self.mv(batch_y, scale)[0, :, -1]), axis=0)
                            pd = np.concatenate((self.mv(input, scale)[0, :, -1], output.detach().cpu().numpy()[0, :, -1]), axis=0)
                            visual(gt, pd, os.path.join(folder_path, str(i) + f'coarse_{scale}.svg'))


        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        result_file_name = 'result.txt'
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.args.prob_forecasting:
            crps_total = np.average(crps_total)
            NLL_total = np.average(NLL_total)
            print(f'crps:{crps_total}, nll:{NLL_total}')
            with open(result_file_name, 'a') as f:
                f.write(f'{setting}  \n gcrps:{crps_total}, nll:{NLL_total} \n \n')
        else:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))
            print(f'running time: {np.array(running_times).sum()}')
            with open(result_file_name, 'a') as f:
                f.write(f'{setting}  \n mse:{mse}, mae:{mae} \n \n')
            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        return
