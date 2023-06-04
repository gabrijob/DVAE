#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import argparse
from tqdm import tqdm
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from dvae.dataset.utils.metric_data_utils import plot_two_curves_graph, plot_n_grids_two_curves_subgraph, plot_n_batches_two_curves_subgraph
from dvae.learning_algo import LearningAlgorithm
from dvae.learning_algo_ss import LearningAlgorithm_ss
from dvae.utils.eval_metric import compute_rmse
from dvae.dataset.offline_metrics_dataset import OfflinePrometheusMetrics
from dvae.utils.loss import loss_KLD

torch.manual_seed(0)
np.random.seed(0)

# TODO: Create class with methods for using the DVAE and returning the needed format (RMSE, AED, KLD, regen, Z,...)
# DataSet and Dataloader not inside this class
# rmse, kld and aed lists are not inside the class either



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
        self.parser.add_argument('--test_dir', type=str, default='/home/gjagrabher/Code/the-prometheus-metrics-dataset/5-minutes-metrics/teastore/teastore-webui', help='test dataset')
        self.parser.add_argument('--anomaly_test_dir', type=str, default='/home/gjagrabher/Code/the-prometheus-metrics-dataset/5-minutes-metrics/teastore/teastore-db', help='anomaly test dataset')
        # Results directory
        self.parser.add_argument('--ret_dir', type=str, default='./data/tmp', help='tmp dir for metric reconstruction')
    def get_params(self):
        self._initial()
        self.opt = self.parser.parse_args()
        params = vars(self.opt)
        return params

def mask_zero():
    """
    Static mask of value 0.
    """
    return 0

def mask_random(size=None):
    """
    Mask of random floats between the half-open interval [0.0, 1.0).
    """
    return np.random.random(size)

def eval_simple(dataloader):

    list_rmse = []
    list_kld = []
    list_aed = []
    for _, batch_data in tqdm(enumerate(dataloader)):

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
    print('RMSE: {:.4f}'.format(np.mean(np_rmse)))
    print('KLD: {:.4f}'.format(np.mean(np_kld)))
    print('AED: {:.4f}'.format(np.mean(np_aed)))


def eval_masking(dataloader, masking_limit, masked_metrics_idxs=[0]):

    rmse_masked_all_t = np.zeros(masking_limit)
    aed_masked_all_t = np.zeros(masking_limit)
    rmse_masked_intern_all_t = np.zeros(masking_limit)
    aed_masked_intern_all_t = np.zeros(masking_limit)

    for t in range(0, masking_limit):
        list_rmse_masked_on_t = []
        list_aed_masked_on_t = []
        list_rmse_intern_masked_on_t = []
        list_aed_intern_masked_on_t = []

        print("Testing masked inputs on t-{}".format(t))
        for _, batch_data in tqdm(enumerate(dataloader)):
            batch_data = batch_data.permute(1, 0, 2)

            masked_batch_data = torch.clone(batch_data)

            if t == 0:
                intern_masked_batch_data = torch.clone(batch_data)
            else:
                masked_batch_data[-t:, :, :] = 0

                intern_masked_batch_data = torch.clone(masked_batch_data)
                for idx in masked_metrics_idxs:
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
        rmse_masked_all_t[t] = np.mean(np_rmse_masked_on_t)
        aed_masked_all_t[t] = np.mean(np_aed_masked_on_t)
        #print('RMSE masked on t-{}: {:.4f}'.format(t, rmse_masked_all_t[t]))
        #print('AED masked on t-{}: {:.4f}'.format(t, aed_masked_all_t[t]))

        np_rmse_intern_masked_on_t = np.array(list_rmse_intern_masked_on_t)
        np_aed_intern_masked_on_t = np.array(list_aed_intern_masked_on_t)
        rmse_masked_intern_all_t[t]= np.mean(np_rmse_intern_masked_on_t)
        aed_masked_intern_all_t[t] = np.mean(np_aed_intern_masked_on_t)
        #print('RMSE masked on t-{} for intern metrics: {:.4f}'.format(t, rmse_masked_intern_all_t[t]))
        #print('AED masked on t-{} for intern metrics: {:.4f}'.format(t, aed_masked_intern_all_t[t]))

    x = range(0, masking_limit)

    plot_two_curves_graph(x, rmse_masked_all_t, rmse_masked_intern_all_t, 
                        't:T masked', 'Avg. RMSE', 'All metrics', 'Load & specs only', 'RMSE masked t:T', 
                        os.path.join(params['ret_dir'], 'evaluation_RMSE_masked_{}.png'.format(learning_algo.model_name)))

    plot_two_curves_graph(x, aed_masked_all_t, aed_masked_intern_all_t, 
                        't:T masked', 'Avg. AED', 'All metrics', 'Load & specs only', 'AED masked t:T', 
                        os.path.join(params['ret_dir'], 'evaluation_AED_masked_{}.png'.format(learning_algo.model_name)))


def eval_generation(dataloader, metrics, batch_size=1):
    # Create eval directory
    eval_dir = os.path.join(params['ret_dir'], 'eval_generation', 'unmasked')
    if not(os.path.isdir(eval_dir)):
                os.makedirs(eval_dir)

    it = iter(dataloader)
    for i in range(0, batch_size):
        batch_data = next(it)
        batch_data = batch_data.permute(1, 0, 2)

        with torch.no_grad():
            recon_batch_data = dvae(batch_data)

        orig_data = batch_data.to('cpu').detach().squeeze().numpy()
        data_recon = recon_batch_data.to('cpu').detach().squeeze().numpy()

        #savepath = os.path.join(eval_dir, 'evaluation_GEN_{}_{}.png'.format(learning_algo.model_name, i))
        #plot_n_grids_two_curves_subgraph(y_orig=np.transpose(orig_data), y_prime=np.transpose(data_recon),
        #                                 rows=3, cols=5, y_names=metrics, savepath=savepath)
        
        # Plot comparison graphs
        """
        orig_input = np.transpose(orig_data)
        generated = np.transpose(data_recon)
        tot_seqs = range(0, orig_input.shape[1]) 
        for m, metric in enumerate(metrics):
            savepath = os.path.join(eval_dir, 'evaluation_GEN_COMP_{}_{}_BATCH_{}.png'.format(learning_algo.model_name, metric, i))
            #plot_two_curves_graph(tot_seqs, orig_input[m, :], generated[m, :], x_label='time(s)', y_label='value',
            #                    y1_label='Original', y2_label='Generated', title=metric,
            #                    savepath=savepath)     
            plot_n_batches_two_curves_subgraph(nb_batches=orig_input.shape[1], seq_len=orig_input.shape[0], line_len=5, 
                                y_orig=orig_input[m, :], y_prime=generated[m, :], savepath=savepath)
        """
        if i==0:
            orig_input = np.transpose(orig_data)
            generated = np.transpose(data_recon)
        else:
            orig_input = np.concatenate((orig_input, np.transpose(orig_data)), axis=1)
            generated = np.concatenate((generated, np.transpose(data_recon)), axis=1)

    tot_seqs = range(0, orig_input.shape[1])
    seq_len = int(orig_input.shape[1]/batch_size)
    # Plot comparison graphs of regenerated inputs
    for i, metric in enumerate(metrics):
        savepath = os.path.join(eval_dir, 'evaluation_GEN_COMP_{}_{}.png'.format(learning_algo.model_name, metric))
        #plot_two_curves_graph(tot_seqs, orig_input[i, :], generated[i, :], x_label='time(s)', y_label='value',
        #                      y1_label='Original', y2_label='Generated', title=metric,
        #                      savepath=savepath)
        plot_n_batches_two_curves_subgraph(nb_batches=batch_size, seq_len=seq_len, line_len=5, title=metric,
                                y_orig=orig_input[i, :], y_prime=generated[i, :], savepath=savepath)    


def eval_generation_masked(dataloader, metrics, batch_size=1, mask_t=1, masked_metrics_idxs=[0]):
    # Create eval directory
    eval_dir = os.path.join(params['ret_dir'], 'eval_generation', 'masked')
    if not(os.path.isdir(eval_dir)):
                os.makedirs(eval_dir)

    it = iter(dataloader)
    for i in range(0, batch_size):
        batch_data = next(it)
        batch_data = batch_data.permute(1, 0, 2)

        masked_batch_data = torch.clone(batch_data)

        if mask_t == 0:
            intern_masked_batch_data = torch.clone(batch_data)
        else:
            mask_shape = list(masked_batch_data[-mask_t:, :, :].size())
            masked_batch_data[-mask_t:, :, :] = torch.from_numpy(mask_random(mask_shape))
            #masked_batch_data[-mask_t:, :, :] = mask_zero()

            intern_masked_batch_data = torch.clone(masked_batch_data)
            for idx in masked_metrics_idxs:
                intern_masked_batch_data[-mask_t:, :, idx] = batch_data[-mask_t:, :, idx]

        with torch.no_grad():
            recon_batch_data = dvae(masked_batch_data)
            intern_recon_batch_data = dvae(intern_masked_batch_data)

        orig_data = batch_data.to('cpu').detach().squeeze().numpy()
        data_recon = recon_batch_data.to('cpu').detach().squeeze().numpy()
        intern_data_recon = intern_recon_batch_data.to('cpu').detach().squeeze().numpy()

        # Plot per sequence graphs
        #savepath = os.path.join(eval_dir, 'eval_GEN_MASK_FULL_{}_T-{}_{}.png'.format(learning_algo.model_name, mask_t, i))
        #plot_n_grids_two_curves_subgraph(y_orig=np.transpose(orig_data), y_prime=np.transpose(data_recon),
        #                                 rows=3, cols=5, y_names=metrics, savepath=savepath)
        #savepath = os.path.join(eval_dir, 'eval_GEN_MASK_PARTIAL_{}_T-{}_{}.png'.format(learning_algo.model_name, mask_t, i))
        #plot_n_grids_two_curves_subgraph(y_orig=np.transpose(orig_data), y_prime=np.transpose(intern_data_recon),
        #                                 rows=3, cols=5, y_names=metrics, savepath=savepath)
        
        if i==0:
            orig_input = np.transpose(orig_data)
            generated_mask_full = np.transpose(data_recon)
            generated_mask_partial = np.transpose(intern_data_recon)
        else:
            orig_input = np.concatenate((orig_input, np.transpose(orig_data)), axis=1)
            generated_mask_full = np.concatenate((generated_mask_full, np.transpose(data_recon)), axis=1)
            generated_mask_partial = np.concatenate((generated_mask_partial, np.transpose(intern_data_recon)), axis=1)

    tot_seqs = range(0, orig_input.shape[1])
    # Plot comparison graphs of regenerated inputs
    mask_full_dir = os.path.join(eval_dir, 'MASK_FULL_{}_T-{}'.format(learning_algo.model_name, mask_t))
    if not(os.path.isdir(mask_full_dir)):
                os.makedirs(mask_full_dir)

    mask_partial_dir = os.path.join(eval_dir, 'MASK_PARTIAL_{}_T-{}'.format(learning_algo.model_name, mask_t))
    if not(os.path.isdir(mask_partial_dir)):
                os.makedirs(mask_partial_dir)

    for i, metric in enumerate(metrics):
        savepath = os.path.join(mask_full_dir, 'eval_GEN_COMP_MASK_FULL_{}_T-{}_{}.png'.format(learning_algo.model_name, mask_t, metric))
        plot_two_curves_graph(tot_seqs, orig_input[i, :], generated_mask_full[i, :], x_label='time(s)', y_label='value',
                              y1_label='Original', y2_label='Generated Full Mask', title=metric,
                              savepath=savepath)
        
        savepath = os.path.join(mask_partial_dir, 'eval_GEN_COMP_MASK_PARTIAL_{}_T-{}_{}.png'.format(learning_algo.model_name, mask_t, metric))
        plot_two_curves_graph(tot_seqs, orig_input[i, :], generated_mask_partial[i, :], x_label='time(s)', y_label='value',
                              y1_label='Original', y2_label='Generated Load&Config Mask', title=metric,
                              savepath=savepath)


def eval_generation_masked_filling_window(dataloader, metrics, batch_size=1, mask_t=1, masked_metrics_idxs=[0]):
    # Create eval directory
    eval_dir = os.path.join(params['ret_dir'], 'eval_generation', 'masked')
    if not(os.path.isdir(eval_dir)):
                os.makedirs(eval_dir)

    it = iter(dataloader)
    for i in range(0, batch_size):
        batch_data = next(it)
        batch_data = batch_data.permute(1, 0, 2)

        # Slice away all of th T-t_mask that should be predicted
        masked_batch_data = torch.clone(batch_data[:-mask_t, :, :])
        intern_masked_batch_data = torch.clone(masked_batch_data[:-mask_t, :, :])

        for i_mask in range(0, mask_t):
            # Append an instant of masked metrics 
            mask_shape = list(masked_batch_data[-1:, :, :].size())
            masked_batch_data = torch.cat((masked_batch_data, torch.rand(mask_shape)), 0)
            #masked_batch_data = torch.cat((masked_batch_data, torch.zeros(mask_shape)), 0)

            intern_masked_batch_data = torch.clone(masked_batch_data)
            for idx in masked_metrics_idxs:
                # Get idx metric from original input
                intern_masked_batch_data[-1, :, idx] = batch_data[-mask_t+i_mask, :, idx]

            with torch.no_grad():
                recon_batch_data = dvae(masked_batch_data)
                intern_recon_batch_data = dvae(intern_masked_batch_data)
                masked_batch_data = recon_batch_data
                intern_masked_batch_data = intern_recon_batch_data

        orig_data = batch_data.to('cpu').detach().squeeze().numpy()
        data_recon = recon_batch_data.to('cpu').detach().squeeze().numpy()
        intern_data_recon = intern_recon_batch_data.to('cpu').detach().squeeze().numpy()

        # Plot comparison graphs
        """
        orig_input = np.transpose(orig_data)
        generated_mask_full = np.transpose(data_recon)
        generated_mask_partial = np.transpose(intern_data_recon)

        tot_seqs = range(0, orig_input.shape[1])      
        mask_full_dir = os.path.join(eval_dir, 'MASK_FULL_{}_T-{}'.format(learning_algo.model_name, mask_t))
        if not(os.path.isdir(mask_full_dir)):
                    os.makedirs(mask_full_dir)

        mask_partial_dir = os.path.join(eval_dir, 'MASK_PARTIAL_{}_T-{}'.format(learning_algo.model_name, mask_t))
        if not(os.path.isdir(mask_partial_dir)):
                    os.makedirs(mask_partial_dir)

        for m, metric in enumerate(metrics):
            savepath = os.path.join(mask_full_dir, 'eval_GEN_COMP_MASK_FULL_FW_{}_T-{}_{}_BATCH_{}.png'.format(learning_algo.model_name, mask_t, metric, i))
            #plot_two_curves_graph(tot_seqs, orig_input[m, :], generated_mask_full[m, :], x_label='time(s)', y_label='value',
            #                    y1_label='Original', y2_label='Generated Full Mask', title=metric,
            #                    savepath=savepath)
                        
            savepath = os.path.join(mask_partial_dir, 'eval_GEN_COMP_MASK_PARTIAL_FW_{}_T-{}_{}_BATCH_{}.png'.format(learning_algo.model_name, mask_t, metric, i))
            #plot_two_curves_graph(tot_seqs, orig_input[m, :], generated_mask_partial[m, :], x_label='time(s)', y_label='value',
            #                    y1_label='Original', y2_label='Generated Load&Config Mask', title=metric,
            #                    savepath=savepath)    
        """
        if i==0:
            orig_input = np.transpose(orig_data)
            generated_mask_full = np.transpose(data_recon)
            generated_mask_partial = np.transpose(intern_data_recon)
        else:
            orig_input = np.concatenate((orig_input, np.transpose(orig_data)), axis=1)
            generated_mask_full = np.concatenate((generated_mask_full, np.transpose(data_recon)), axis=1)
            generated_mask_partial = np.concatenate((generated_mask_partial, np.transpose(intern_data_recon)), axis=1)

    tot_seqs = range(0, orig_input.shape[1])
    seq_len = int(orig_input.shape[1]/batch_size)
    # Plot comparison graphs of regenerated inputs
    mask_full_dir = os.path.join(eval_dir, 'MASK_FULL_{}_T-{}'.format(learning_algo.model_name, mask_t))
    if not(os.path.isdir(mask_full_dir)):
                os.makedirs(mask_full_dir)

    mask_partial_dir = os.path.join(eval_dir, 'MASK_PARTIAL_{}_T-{}'.format(learning_algo.model_name, mask_t))
    if not(os.path.isdir(mask_partial_dir)):
                os.makedirs(mask_partial_dir)

    for i, metric in enumerate(metrics):
        savepath = os.path.join(mask_full_dir, 'eval_GEN_COMP_MASK_FULL_{}_T-{}_{}.png'.format(learning_algo.model_name, mask_t, metric))
        #plot_two_curves_graph(tot_seqs, orig_input[i, :], generated_mask_full[i, :], x_label='time(s)', y_label='value',
        #                      y1_label='Original', y2_label='Generated Full Mask', title=metric,
        #                      savepath=savepath)
        plot_n_batches_two_curves_subgraph(nb_batches=batch_size, seq_len=seq_len, line_len=5, title=metric,
                                y_orig=orig_input[i, :], y_prime=generated_mask_full[i, :], savepath=savepath) 

        savepath = os.path.join(mask_partial_dir, 'eval_GEN_COMP_MASK_PARTIAL_{}_T-{}_{}.png'.format(learning_algo.model_name, mask_t, metric))
        #plot_two_curves_graph(tot_seqs, orig_input[i, :], generated_mask_partial[i, :], x_label='time(s)', y_label='value',
        #                      y1_label='Original', y2_label='Generated Load&Config Mask', title=metric,
        #                      savepath=savepath)
        plot_n_batches_two_curves_subgraph(nb_batches=batch_size, seq_len=seq_len, line_len=5, title=metric,
                                y_orig=orig_input[i, :], y_prime=generated_mask_partial[i, :], savepath=savepath) 


def eval_generation_masked_sliding_window(dataloader, metrics, batch_size=1, mask_t=1, masked_metrics_idxs=[0]):
  # Create eval directory
    eval_dir = os.path.join(params['ret_dir'], 'eval_generation', 'masked')
    if not(os.path.isdir(eval_dir)):
                os.makedirs(eval_dir)

    it = iter(dataloader)
    for i in range(0, batch_size, 2):
        batch_data_1 = next(it)
        batch_data_1 = batch_data_1.permute(1, 0, 2)
        batch_data_2 = next(it)
        batch_data_2 = batch_data_2.permute(1, 0, 2)

        # Slice away all of th T-t_mask that should be predicted
        masked_batch_data = torch.clone(batch_data_1[:-mask_t, :, :])
        intern_masked_batch_data = torch.clone(masked_batch_data[:-mask_t, :, :])

        for i_mask in range(0, mask_t):
            # Append an instant of masked metrics 
            mask_shape = list(masked_batch_data[-1:, :, :].size())
            masked_batch_data = torch.cat((masked_batch_data, torch.rand(mask_shape)), 0)
            #masked_batch_data = torch.cat((masked_batch_data, torch.zeros(mask_shape)), 0)

            intern_masked_batch_data = torch.clone(masked_batch_data)
            for idx in masked_metrics_idxs:
                # Get idx metric from original input
                intern_masked_batch_data[-1, :, idx] = batch_data_1[-mask_t+i_mask, :, idx]

            with torch.no_grad():
                recon_batch_data = dvae(masked_batch_data)
                intern_recon_batch_data = dvae(intern_masked_batch_data)
                masked_batch_data = recon_batch_data
                intern_masked_batch_data = intern_recon_batch_data

        orig_data = batch_data_1.to('cpu').detach().squeeze().numpy()
        data_recon = recon_batch_data.to('cpu').detach().squeeze().numpy()
        intern_data_recon = intern_recon_batch_data.to('cpu').detach().squeeze().numpy()

        if i==0:
            orig_input = np.transpose(orig_data)
            generated_mask_full = np.transpose(data_recon)
            generated_mask_partial = np.transpose(intern_data_recon)
        else:
            orig_input = np.concatenate((orig_input, np.transpose(orig_data)), axis=1)
            generated_mask_full = np.concatenate((generated_mask_full, np.transpose(data_recon)), axis=1)
            generated_mask_partial = np.concatenate((generated_mask_partial, np.transpose(intern_data_recon)), axis=1)

    tot_seqs = range(0, orig_input.shape[1])
    # Plot comparison graphs of regenerated inputs
    mask_full_dir = os.path.join(eval_dir, 'MASK_FULL_{}_T-{}'.format(learning_algo.model_name, mask_t))
    if not(os.path.isdir(mask_full_dir)):
                os.makedirs(mask_full_dir)

    mask_partial_dir = os.path.join(eval_dir, 'MASK_PARTIAL_{}_T-{}'.format(learning_algo.model_name, mask_t))
    if not(os.path.isdir(mask_partial_dir)):
                os.makedirs(mask_partial_dir)

    for i, metric in enumerate(metrics):
        savepath = os.path.join(mask_full_dir, 'eval_GEN_COMP_MASK_FULL_FW_{}_T-{}_{}.png'.format(learning_algo.model_name, mask_t, metric))
        plot_two_curves_graph(tot_seqs, orig_input[i, :], generated_mask_full[i, :], x_label='time(s)', y_label='value',
                              y1_label='Original', y2_label='Generated Full Mask', title=metric,
                              savepath=savepath)
        
        savepath = os.path.join(mask_partial_dir, 'eval_GEN_COMP_MASK_PARTIAL_FW_{}_T-{}_{}.png'.format(learning_algo.model_name, mask_t, metric))
        plot_two_curves_graph(tot_seqs, orig_input[i, :], generated_mask_partial[i, :], x_label='time(s)', y_label='value',
                              y1_label='Original', y2_label='Generated Load&Config Mask', title=metric,
                              savepath=savepath)



if __name__ == '__main__':

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


    test_dataset = OfflinePrometheusMetrics(svc_datadir=params['test_dir'], shuffle=False, seq_len=60, split=1, dist_policy='function')
    test_num = test_dataset.__len__()
    seq_len = test_dataset.__seq_len__()
    metrics = test_dataset.__metric_names__()
    print('Test samples: {}'.format(test_num))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    #eval_masking(dataloader=test_dataloader, masking_limit=seq_len, masked_metrics_idxs=[6, 9, 10, 11, 12, 13, 14])
    #eval_simple(dataloader=test_dataloader)
    #eval_generation(dataloader=test_dataloader, metrics=metrics, batch_size=20)
    percentage_masked = 0.1
    masked_t = int(seq_len*percentage_masked)
    #eval_generation_masked(dataloader=test_dataloader, metrics=metrics, batch_size=3, mask_t=1, masked_metrics_idxs=[3])
    eval_generation_masked_filling_window(dataloader=test_dataloader, metrics=metrics, batch_size=20, mask_t=masked_t, masked_metrics_idxs=[3])
