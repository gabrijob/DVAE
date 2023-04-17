#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import argparse
from tqdm import tqdm
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from dvae.learning_algo import LearningAlgorithm
from dvae.learning_algo_ss import LearningAlgorithm_ss
from dvae.utils.eval_metric import compute_rmse
from dvae.dataset.offline_metrics_dataset import OfflinePrometheusMetrics
from dvae.utils.loss import loss_KLD

torch.manual_seed(0)
np.random.seed(0)

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # Basic config file
        self.parser.add_argument('--model', action='store_true', help='schedule sampling')
        self.parser.add_argument('--ss', action='store_true', help='schedule sampling')
        self.parser.add_argument('--cfg', type=str, default=None, help='config path')
        self.parser.add_argument('--saved_dict', type=str, default=None, help='trained model dict')
        # Dataset
        self.parser.add_argument('--test_dir', type=str, default='./data/TeaMe_dataset_d15/teastore-webui', help='test dataset')
        self.parser.add_argument('--anomaly_test_dir', type=str, default='./data/TeaMe_dataset_d18/anomaly-teastore/teastore-db', help='anomaly test dataset')
        # Results directory
        self.parser.add_argument('--ret_dir', type=str, default='./data/tmp', help='tmp dir for metric reconstruction')
    def get_params(self):
        self._initial()
        self.opt = self.parser.parse_args()
        params = vars(self.opt)
        return params

params = Options().get_params()

if params['ss']:
    learning_algo = LearningAlgorithm_ss(params=params)
else:
    learning_algo = LearningAlgorithm(params=params)
learning_algo.build_model()
dvae = learning_algo.model
dvae.load_state_dict(torch.load(params['saved_dict'], map_location='cpu'))
#eval_metrics = EvalMetrics(metric='all')
dvae.eval()
cfg = learning_algo.cfg
print('Total params: %.2fM' % (sum(p.numel() for p in dvae.parameters()) / 1000000.0))


test_dataset = OfflinePrometheusMetrics(svc_datadir=params['test_dir'], shuffle=True, seq_len=30, split=1)
test_num = test_dataset.__len__()
print('Test samples: {}'.format(test_num))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)

RMSE = 0
KLD = 0
AED = 0
list_rmse = []
list_kld = []
list_aed = []
for _, batch_data in tqdm(enumerate(test_dataloader)):

    #batch_data = batch_data.to('cuda')
    batch_data = batch_data.permute(1, 0, 2)
    with torch.no_grad():
        recon_batch_data = dvae(batch_data)

    orig_data = batch_data.to('cpu').detach().squeeze().numpy()
    data_recon = recon_batch_data.to('cpu').detach().squeeze().numpy()

    #loss_recon = compute_rmse(orig_data, data_recon)
    RMSE = np.sqrt(np.square(orig_data - data_recon).mean())
    
    seq_len = orig_data.shape[0]
    if learning_algo.model_name == 'DSAE':
        loss_kl_z = loss_KLD(dvae.z_mean, dvae.z_logvar, dvae.z_mean_p, dvae.z_logvar_p)
        loss_kl_v = loss_KLD(dvae.v_mean, dvae.v_logvar, dvae.v_mean_p, dvae.v_logvar_p)
        loss_kl = loss_kl_z + loss_kl_v
    else:
        loss_kl = loss_KLD(dvae.z_mean, dvae.z_logvar, dvae.z_mean_p, dvae.z_logvar_p)
    KLD = loss_kl / seq_len

    #AED = np.sqrt(np.square(orig_data - data_recon))
    AED = np.linalg.norm(orig_data-data_recon)

    list_rmse.append(RMSE)
    list_kld.append(KLD)
    list_aed.append(AED)

np_rmse = np.array(list_rmse)
np_kld = np.array(list_kld)
np_aed = np.array(list_aed)

#print(np_rmse)
#print(np_kld)
print('RMSE: {:.4f}'.format(np.mean(np_rmse)))
print('KLD: {:.4f}'.format(np.mean(np_kld)))
print('AED: {:.4f}'.format(np.mean(np_aed)))


plt.clf()
fig = plt.figure(figsize=(8,6))
plt.rcParams['font.size'] = 12
#plt.plot(np_rmse, label='Evaluation')
plt.violinplot(np_rmse, showmedians=True)
plt.xticks([1], ['RMSE'])
#plt.legend(fontsize=16, title='{}: Root Mean Square Error'.format(learning_algo.model_name), title_fontsize=20)
#plt.xlabel('seq', fontdict={'size':16})
plt.ylabel('loss', fontdict={'size':16})
fig_file = os.path.join(params['ret_dir'], 'evaluation_RMSE_{}.png'.format(learning_algo.model_name))
plt.savefig(fig_file) 


plt.clf()
fig = plt.figure(figsize=(8,6))
plt.rcParams['font.size'] = 12
#plt.plot(np_kld, label='Evaluation')
plt.violinplot(np_kld, showmedians=True)
plt.xticks([1], ['KLD'])
#plt.legend(fontsize=16, title='{}: KL Divergence'.format(learning_algo.model_name), title_fontsize=20)
#plt.xlabel('seq', fontdict={'size':16})
plt.ylabel('loss', fontdict={'size':16})
fig_file = os.path.join(params['ret_dir'], 'evaluation_KLD_{}.png'.format(learning_algo.model_name))
plt.savefig(fig_file)

plt.clf()
fig = plt.figure(figsize=(8,6))
plt.rcParams['font.size'] = 12
#plt.plot(np_aed, label='Evaluation')
plt.violinplot(np_aed, showmedians=True)
plt.xticks([1], ['AED'])
#plt.legend(fontsize=16, title='{}: Euclidian Disitance'.format(learning_algo.model_name), title_fontsize=20)
#plt.xlabel('seq', fontdict={'size':16})
plt.ylabel('loss', fontdict={'size':16})
fig_file = os.path.join(params['ret_dir'], 'evaluation_AED_{}.png'.format(learning_algo.model_name))
plt.savefig(fig_file)


list_rmse_masked_on_t = []
list_aed_masked_on_t = []
list_rmse_intern_masked_on_t = []
list_aed_intern_masked_on_t = []
t = 5
network_metrics_idxs = [5, 6]
print("Testing masked inputs on t-{}".format(t))
for _, batch_data in tqdm(enumerate(test_dataloader)):
    batch_data = batch_data.permute(1, 0, 2)

    masked_batch_data = torch.clone(batch_data)
    masked_batch_data[-t:, :, :] = 0

    intern_masked_batch_data = torch.clone(masked_batch_data)
    for idx in network_metrics_idxs:
        intern_masked_batch_data[-t:, :, idx] = batch_data[-t:, :, idx]

    with torch.no_grad():
        recon_batch_data = dvae(masked_batch_data)
        intern_recon_batch_data = dvae(intern_masked_batch_data)

    orig_data = batch_data.to('cpu').detach().squeeze().numpy()
    data_recon = recon_batch_data.to('cpu').detach().squeeze().numpy()
    intern_data_recon = intern_recon_batch_data.to('cpu').detach().squeeze().numpy()

    RMSE = np.sqrt(np.square(orig_data - data_recon).mean())
    AED = np.linalg.norm(orig_data-data_recon)

    list_rmse_masked_on_t.append(RMSE)
    list_aed_masked_on_t.append(AED)

    RMSE = np.sqrt(np.square(orig_data - intern_data_recon).mean())
    AED = np.linalg.norm(orig_data - intern_data_recon)

    list_rmse_intern_masked_on_t.append(RMSE)
    list_aed_intern_masked_on_t.append(AED)


np_rmse_masked_on_t = np.array(list_rmse_masked_on_t)
np_aed_masked_on_t = np.array(list_aed_masked_on_t)
print('RMSE masked on t-{}: {:.4f}'.format(t, np.mean(np_rmse_masked_on_t)))
print('AED masked on t-{}: {:.4f}'.format(t, np.mean(np_aed_masked_on_t)))
np_rmse_intern_masked_on_t = np.array(list_rmse_intern_masked_on_t)
np_aed_intern_masked_on_t = np.array(list_aed_intern_masked_on_t)
print('RMSE masked on t-{} for intern metrics: {:.4f}'.format(t, np.mean(np_rmse_intern_masked_on_t)))
print('AED masked on t-{} for intern metrics: {:.4f}'.format(t, np.mean(np_aed_intern_masked_on_t)))

"""
anomaly_test_dataset = OfflinePrometheusMetrics(svc_datadir=params['anomaly_test_dir'], shuffle=True, seq_len=30, split=1)
test_num = anomaly_test_dataset.__len__()
print('Anomaly Test samples: {}'.format(test_num))
anomaly_test_dataloader = torch.utils.data.DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True, num_workers=2)


list_rmse_anomaly = []
list_aed_anomaly = []
list_kld_anomaly = []
print("Anomaly Metrics Testing")
for _, batch_data in tqdm(enumerate(anomaly_test_dataloader)):

    #batch_data = batch_data.to('cuda')
    batch_data = batch_data.permute(1, 0, 2)
    with torch.no_grad():
        recon_batch_data = dvae(batch_data)

    orig_data = batch_data.to('cpu').detach().squeeze().numpy()
    data_recon = recon_batch_data.to('cpu').detach().squeeze().numpy()

    RMSE = np.sqrt(np.square(orig_data - data_recon).mean())
    AED = np.linalg.norm(orig_data - data_recon)

    seq_len = orig_data.shape[0]
    if learning_algo.model_name == 'DSAE':
        loss_kl_z = loss_KLD(dvae.z_mean, dvae.z_logvar, dvae.z_mean_p, dvae.z_logvar_p)
        loss_kl_v = loss_KLD(dvae.v_mean, dvae.v_logvar, dvae.v_mean_p, dvae.v_logvar_p)
        loss_kl = loss_kl_z + loss_kl_v
    else:
        loss_kl = loss_KLD(dvae.z_mean, dvae.z_logvar, dvae.z_mean_p, dvae.z_logvar_p)
    KLD = loss_kl / seq_len

    list_rmse_anomaly.append(RMSE)
    list_kld_anomaly.append(KLD)
    list_aed_anomaly.append(AED)

np_rmse_anomaly = np.array(list_rmse_anomaly)
np_aed_anomaly = np.array(list_aed_anomaly)
np_kld_anomaly = np.array(list_kld_anomaly)
print('RMSE anomaly: {:.4f}'.format(np.mean(np_rmse_anomaly)))
print('KLD anomaly: {:.4f}'.format(np.mean(np_kld_anomaly)))
print('AED anomaly: {:.4f}'.format(np.mean(np_aed_anomaly)))
"""

# Create class with methods for using the DVAE and returning the needed format (RMSE, AED, KLD, regen, Z,...)
# DataSet and Dataloader not inside this class
# rmse, kld and aed lists are not inside the class either