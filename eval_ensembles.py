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
from dvae.dataset.utils.metric_data_utils import plot_two_curves_graph, plot_n_curves, plot_n_batches_two_curves_subgraph, plot_n_batches_four_curves_subgraph
from dvae.learning_algo_ensemble import LearningAlgorithm_ensemble
from dvae.utils.eval_metric import compute_rmse
from dvae.dataset.ensemble_dataset import EnsembleMetrics
from dvae.utils.loss import loss_KLD, loss_PIQD

torch.manual_seed(0)
np.random.seed(0)


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # Basic config file
        self.parser.add_argument('--model', action='store_true', help='schedule sampling')
        self.parser.add_argument('--cfg', type=str, default=None, help='config path')
        self.parser.add_argument('--saved_dict', type=str, default=None, help='trained model dict')
        self.parser.add_argument('--test_target', type=str, default='sin', help='name of the test target dir')
        # Dataset
        self.parser.add_argument('--test_dir', type=str, default='/home/gjagrabher/Code/the-prometheus-metrics-dataset/5-minutes-metrics/teastore/teastore-webui/node_dist_1/hw_spec_2/pod_spec_1/teastore_browse/', help='test dataset')
        # Results directory
        self.parser.add_argument('--ret_dir', type=str, default='./data/tmp', help='tmp dir for metric reconstruction')
    def get_params(self):
        self._initial()
        self.opt = self.parser.parse_args()
        params = vars(self.opt)
        return params


def apply_full_mask(original_data, mask_t=1, mask_type='rand'):
    """
    Replaces the last 'mask_t' values of tensor 'original_data' with 'mask_type' values.
    Where 'mask_type' can be either random or zero values.
    """
    if mask_t < 1:
        return original_data
    
    # Copy from original data only the data which won't be masked
    masked_data = torch.clone(original_data[:-mask_t, :, :])

    # Append mask to copied data
    mask_shape = list(masked_data[-mask_t:, :, :].size())
    if mask_type == 'rand':
        masked_data = torch.cat((masked_data, torch.rand(mask_shape)), 0)
    elif mask_type == 'zero':
        masked_data = torch.cat((masked_data, torch.zeros(mask_shape)), 0)
    elif mask_type == 'last':
        masked_data = torch.cat((masked_data, masked_data[-mask_t:, :, :]), 0)
    else: 
        masked_data = torch.cat((masked_data, torch.rand(mask_shape)), 0)

    return masked_data


def apply_partial_mask(original_data, mask_t=1, mask_type='rand', not_masked_idxs=[0]):
    """
    Replaces the last 'mask_t' values of tensor 'original_data' with 'mask_type' values
    only for the metrics not in the 'not_masked_idxs' array.
    Where 'mask_type' can be either random or zero values.
    """
    if mask_t < 1:
        return original_data

    # Copy from original data only the data which won't be masked
    partial_masked_data = torch.clone(original_data[:-mask_t, :, :])

    # Append mask to copied data
    mask_shape = list(partial_masked_data[-mask_t:, :, :].size())
    if mask_type == 'rand':
        partial_masked_data = torch.cat((partial_masked_data, torch.rand(mask_shape)), 0)
    elif mask_type == 'zero':
        partial_masked_data = torch.cat((partial_masked_data, torch.zeros(mask_shape)), 0)
    elif mask_type == 'last':
        partial_masked_data = torch.cat((partial_masked_data, partial_masked_data[-mask_t:, :, :]), 0)
    else: 
        partial_masked_data = torch.cat((partial_masked_data, torch.rand(mask_shape)), 0)

    # Switch non-masked metrics to its original values
    for idx in not_masked_idxs:
        partial_masked_data[-mask_t:, :, idx] = original_data[-mask_t:, :, idx]

    return partial_masked_data


def plot_comparison_graphs(original_data, generated_data, metrics, title='MASK', mask_t=1, batch_size=1, seq_len=10):
    # Create eval directory
    eval_dir = os.path.join(params['ret_dir'], 'eval_generation', 'masked')
    if not(os.path.isdir(eval_dir)):
                os.makedirs(eval_dir)
    
    # Plot comparison graphs of regenerated inputs
    mask_dir = os.path.join(eval_dir, '{}_{}_T-{}'.format(title, learning_algo.model_name, mask_t))
    if not(os.path.isdir(mask_dir)):
                os.makedirs(mask_dir)

    tot_seqs = range(0, original_data.shape[1])
    for i, metric in enumerate(metrics):
        savepath = os.path.join(mask_dir, 'eval_GEN_COMP_{}_{}_T-{}_{}.png'.format(title, learning_algo.model_name, mask_t, metric))
        plot_n_batches_two_curves_subgraph(nb_batches=batch_size, seq_len=seq_len, line_len=5, title=metric,
                                y_orig=original_data[i, :], y_prime=generated_data[i, :], savepath=savepath) 
        savepath = os.path.join(mask_dir, 'single_eval_GEN_COMP_{}_{}_T-{}_{}.png'.format(title, learning_algo.model_name, mask_t, metric))
        plot_two_curves_graph(tot_seqs, original_data[i, :], generated_data[i, :], x_label='time(s)', y_label='value',
                              y1_label='Original', y2_label='Generated with Mask', title=metric,
                              savepath=savepath)


def post_delta_correction(original_data, generated_data, gen_start_idx=0):
    delta_arr = original_data - generated_data

    avg_delta = np.mean(delta_arr, axis=1)
    
    corrected = generated_data + avg_delta[:, np.newaxis] 
    return np.concatenate((original_data[:,:gen_start_idx], corrected[:, gen_start_idx:]), axis=1)

     
def eval_qd(dataloader, metrics, batch_size=1):
    # Create eval directory
    eval_dir = os.path.join(params['ret_dir'], 'eval_generation', 'bounds')
    if not(os.path.isdir(eval_dir)):
                os.makedirs(eval_dir)

    list_qd = []
    list_MPIW = []
    list_PICP = []
    it = iter(dataloader)
    for i in range(0, batch_size):
        batch_data = next(it)
        batch_data = batch_data.permute(1, 0, 2)

        mean_recons = []
        mean_lower_bounds = []
        mean_upper_bounds = []
        for model in ensemble:
            with torch.no_grad():
                recon_batch_data = model(batch_data)

            mean_recons.append(recon_batch_data)
            mean_lower_bounds.append(model.y_lower_bound)
            mean_upper_bounds.append(model.y_upper_bound)
        
        mean_recons = torch.stack(mean_recons, dim=0)
        mean_recons = torch.mean(mean_recons, dim=0)
        mean_lower_bounds = torch.stack(mean_lower_bounds, dim=0)
        mean_lower_bounds = torch.mean(mean_lower_bounds, dim=0)
        mean_upper_bounds = torch.stack(mean_upper_bounds, dim=0)
        mean_upper_bounds = torch.mean(mean_upper_bounds, dim=0)

        orig_data = batch_data.to('cpu').detach().squeeze().numpy()
        data_recon = mean_recons.to('cpu').detach().squeeze().numpy()
        lower_bounds = mean_lower_bounds.to('cpu').detach().squeeze().numpy()
        upper_bounds = mean_upper_bounds.to('cpu').detach().squeeze().numpy()

        QD, MPIW, PICP = loss_PIQD(batch_data, mean_lower_bounds, mean_upper_bounds, alpha=0.2)
        list_qd.append(QD.item())
        list_MPIW.append(MPIW.item())
        list_PICP.append(PICP.item())

        if i==0:
            orig_input = np.transpose(orig_data)
            generated = np.transpose(data_recon)
            gen_lower = np.transpose(lower_bounds)
            gen_upper = np.transpose(upper_bounds)
        else:
            orig_input = np.concatenate((orig_input, np.transpose(orig_data)), axis=1)
            generated = np.concatenate((generated, np.transpose(data_recon)), axis=1)
            gen_lower = np.concatenate((gen_lower, np.transpose(lower_bounds)), axis=1)
            gen_upper = np.concatenate((gen_upper, np.transpose(upper_bounds)), axis=1)

    np_qd = np.array(list_qd)
    print('QD: {:.4f}'.format(np.mean(np_qd)))
    np_mpiw = np.array(list_MPIW)
    print('MPIW: {:.4f}'.format(np.mean(np_mpiw)))
    np_picp = np.array(list_PICP)
    print('PICP: {:.4f}'.format(np.mean(np_picp)))

    tot_seqs = range(0, orig_input.shape[1])
    comp_labels = ['Ground truth', 'Generated', 'Lower bound', 'Upper bound']
    for i, metric in enumerate(metrics):
        comp_arrays = [orig_input[i,:], generated[i,:], gen_lower[i,:], gen_upper[i,:]]
        savepath = os.path.join(eval_dir, 'evaluation_BOUNDS_{}_{}.png'.format(learning_algo.model_name, metric))
        #plot_n_curves(x_arr=tot_seqs, y_arrs=comp_arrays, labels=comp_labels, 
        #            x_label='s (seconds)', y_label='y', title= 'Lower & Upper bound', savepath=savepath)
        
        plot_n_batches_four_curves_subgraph(nb_batches=batch_size, seq_len=seq_len, line_len=5, 
                                y_arrs=comp_arrays, labels=comp_labels, savepath=savepath)


def eval_generation_masked_filling_window(dataloader, metrics, batch_size=1, mask_t=1, masked_metrics_idxs=[0], post_correc=False, mask_comp=False):

    it = iter(dataloader)
    for i in range(0, batch_size):
        batch_data = next(it)
        batch_data = batch_data.permute(1, 0, 2)

        # Slice away all of th T-t_mask that should be generated/predicted
        full_mask_data = torch.clone(batch_data[:-mask_t, :, :])
        partial_mask_data = torch.clone(batch_data[:-mask_t, :, :])
        only_mask_data = torch.clone(batch_data[:-mask_t, :, :])

        for i_mask in range(0, mask_t):
            # Append an instant of the original data to be masked
            t_to_predict = - mask_t + i_mask
            full_mask_data = torch.cat((full_mask_data, batch_data[t_to_predict-1:t_to_predict, :, :]), 0)
            partial_mask_data = torch.cat((partial_mask_data, batch_data[t_to_predict-1:t_to_predict, :, :]), 0)
            only_mask_data = torch.cat((only_mask_data, batch_data[t_to_predict-1:t_to_predict, :, :]), 0)

            # Mask instant t+1 to be generated/predicted
            full_mask_data = apply_full_mask(full_mask_data, mask_t=1, mask_type='last')
            only_mask_data = apply_partial_mask(only_mask_data, mask_t=1, not_masked_idxs=masked_metrics_idxs, mask_type='last')
            partial_mask_data = apply_partial_mask(partial_mask_data, mask_t=1, not_masked_idxs=masked_metrics_idxs, mask_type='last')

            # Generate instant t+1
            with torch.no_grad():
                recon_batch_data = dvae(full_mask_data)
                intern_recon_batch_data = dvae(partial_mask_data)
                # Update only the value of instanst t+1 (the rest should follow the original batch data)
                full_mask_data[-1, :, :] = recon_batch_data[-1, :, :]
                partial_mask_data[-1, :, :] = intern_recon_batch_data[-1, :, :]
                for idx in masked_metrics_idxs:
                    partial_mask_data[-1, :, idx] = batch_data[t_to_predict, :, idx]

        if mask_comp:
            orig_data = only_mask_data.to('cpu').detach().squeeze().numpy()
        else:
            orig_data = batch_data.to('cpu').detach().squeeze().numpy()
        data_recon = full_mask_data.to('cpu').detach().squeeze().numpy()
        intern_data_recon = partial_mask_data.to('cpu').detach().squeeze().numpy()

        orig_input_i = np.transpose(orig_data)
        generated_mask_full_i = np.transpose(data_recon)
        generated_mask_partial_i = np.transpose(intern_data_recon)
        
        # Apply post processing corrections if needed
        if post_correc:
            generated_mask_full_i = post_delta_correction(orig_input_i, generated_mask_full_i, gen_start_idx=window_size)
            generated_mask_partial_i = post_delta_correction(orig_input_i, generated_mask_partial_i, gen_start_idx=window_size)

        if i==0:
            orig_input = orig_input_i
            generated_mask_full = generated_mask_full_i
            generated_mask_partial = generated_mask_partial_i
        else:
            orig_input = np.concatenate((orig_input, orig_input_i), axis=1)
            generated_mask_full = np.concatenate((generated_mask_full, generated_mask_full_i), axis=1)
            generated_mask_partial = np.concatenate((generated_mask_partial, generated_mask_partial_i), axis=1)

    # Plot comparison graphs
    seq_len = int(orig_input.shape[1]/batch_size)
    plot_comparison_graphs(orig_input, generated_mask_full, metrics, 'MASK_FULL', mask_t=mask_t, batch_size=batch_size, seq_len=seq_len)
    plot_comparison_graphs(orig_input, generated_mask_partial, metrics, 'MASK_PARTIAL', mask_t=mask_t, batch_size=batch_size, seq_len=seq_len)


def eval_generation_masked_sliding_window(dataloader, metrics, batch_size=1, seq_len=30, window_size=10, masked_metrics_idxs=[0], post_correc=False, mask_comp=False):
    # window_size = seq_len - mask_t 

    # Evaluate each batch
    it = iter(dataloader)
    for i in range(0, batch_size):
        batch_data = next(it)
        batch_data = batch_data.permute(1, 0, 2)

        # Slice away all the window size that should be static
        full_predicted_data = torch.clone(batch_data[:window_size-1, :, :])
        partial_predicted_data = torch.clone(batch_data[:window_size-1, :, :])
        only_mask_data = torch.clone(batch_data[:window_size-1, :, :])

        for j in range(0, seq_len-window_size+1):
            # Mask instant t+1 to be generated/predicted
            full_mask_data = apply_full_mask(batch_data[j:j+window_size,:,:], mask_t=1, mask_type='last') 
            partial_mask_data = apply_partial_mask(batch_data[j:j+window_size,:,:], mask_t=1, not_masked_idxs=masked_metrics_idxs, mask_type='last')
            only_mask_data = torch.cat((only_mask_data, full_mask_data[-1:, :, :]), 0)

            with torch.no_grad():
                recon_batch_data = dvae(full_mask_data)
                intern_recon_batch_data = dvae(partial_mask_data)
                full_predicted_data = torch.cat((full_predicted_data, recon_batch_data[-1:, :, :]), 0)
                partial_predicted_data = torch.cat((partial_predicted_data, intern_recon_batch_data[-1:, :, :]), 0)
                for idx in masked_metrics_idxs:
                    partial_predicted_data[-1, :, idx] = batch_data[j+window_size-1, :, idx]

        if mask_comp:
            orig_data = only_mask_data.to('cpu').detach().squeeze().numpy()
        else:
            orig_data = batch_data.to('cpu').detach().squeeze().numpy()
        data_recon = full_predicted_data.to('cpu').detach().squeeze().numpy()
        intern_data_recon = partial_predicted_data.to('cpu').detach().squeeze().numpy()

        orig_input_i = np.transpose(orig_data)
        generated_mask_full_i = np.transpose(data_recon)
        generated_mask_partial_i = np.transpose(intern_data_recon)
        
        # Apply post processing corrections if needed
        if post_correc:
            generated_mask_full_i = post_delta_correction(orig_input_i, generated_mask_full_i, gen_start_idx=window_size)
            generated_mask_partial_i = post_delta_correction(orig_input_i, generated_mask_partial_i, gen_start_idx=window_size)

        if i==0:
            orig_input = orig_input_i
            generated_mask_full = generated_mask_full_i
            generated_mask_partial = generated_mask_partial_i
        else:
            orig_input = np.concatenate((orig_input, orig_input_i), axis=1)
            generated_mask_full = np.concatenate((generated_mask_full, generated_mask_full_i), axis=1)
            generated_mask_partial = np.concatenate((generated_mask_partial, generated_mask_partial_i), axis=1)

    # Plot comparison graphs of regenerated inputs
    plot_comparison_graphs(orig_input, generated_mask_full, metrics, 'MASK_FULL', mask_t=seq_len-window_size, batch_size=batch_size, seq_len=seq_len)
    plot_comparison_graphs(orig_input, generated_mask_partial, metrics, 'MASK_PARTIAL', mask_t=seq_len-window_size, batch_size=batch_size, seq_len=seq_len)


if __name__ == '__main__':

    params = Options().get_params()
    dirs = os.listdir(params['saved_dict'])
    ensemble_dirs = [dirname for dirname in dirs if not dirname.startswith(params['test_target'])]
    test_target_dir = [dirname for dirname in dirs if dirname.startswith(params['test_target'])]
    
    ensemble = []
    for single_dir in ensemble_dirs: 
        learning_algo = LearningAlgorithm_ensemble(params=params)
        learning_algo.build_model()
        model = learning_algo.curr_model
        dict_path = os.path.join(params['saved_dict'], single_dir, 'SRNN_final_epoch.pt')
        model.load_state_dict(torch.load(dict_path, map_location='cpu'))
        model.eval()
        ensemble.append(model)


    test_dataset = EnsembleMetrics(datadir=params['test_dir'], 
                        shuffle=False, seq_len=300, split=1, 
                        specific=test_target_dir[0])
    test_num = test_dataset.__len__()
    seq_len = test_dataset.__seq_len__()
    metrics = test_dataset.__metric_names__()
    print('Test samples: {}'.format(test_num))
    print('Ensemble: {}'.format(ensemble_dirs))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Run evaluation
    eval_qd(dataloader=test_dataloader, metrics=metrics, batch_size=10)
    #eval_generation_masked_filling_window(dataloader=test_dataloader, metrics=metrics, batch_size=30, mask_t=10, masked_metrics_idxs=[3], mask_comp=False)
    #eval_generation_masked_sliding_window(dataloader=test_dataloader, metrics=metrics, batch_size=20, seq_len=seq_len, window_size=30, masked_metrics_idxs=[3], post_correc=False, mask_comp=False)
