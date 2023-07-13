#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset of Prometheus metrics for offline learning.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
import random
import math
from sklearn import preprocessing

# Use 3 as a random seed to keep consistency across experiments
SEED = 1


# Get JSON (or other) files with the results of metrics queries from several days/workload sessions
# Online training will come from http requesting at runtime (later)
def build_dataloader(cfg, specific):
    
    # Load config params
    data_dir = cfg.get('User', 'data_dir')
    shuffle = cfg.getboolean('DataFrame', 'shuffle')
    batch_size = cfg.getint('DataFrame', 'batch_size')
    num_workers = cfg.getint('DataFrame', 'num_workers')
    sequence_len = cfg.getint('DataFrame', 'sequence_len')

    # Training dataset
    split = 0 # train
    train_dataset = EnsembleMetrics(datadir=data_dir, seq_len=sequence_len, shuffle=shuffle, split=split, specific=specific)
    split = 2 # validation
    val_dataset = EnsembleMetrics(datadir=data_dir, seq_len=sequence_len, shuffle=shuffle, split=split, specific=specific)
    train_num = train_dataset.__len__()
    val_num = val_dataset.__len__()
    
    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                shuffle=shuffle, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                shuffle=shuffle, num_workers=num_workers)

    return train_dataloader, val_dataloader, train_num, val_num

    

class EnsembleMetrics(Dataset):
    def __init__(self, datadir, seq_len, shuffle, split=0, removed_idx=[], specific=""):
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.split = split
        split_name = ['TRAINING', 'TEST', 'VALIDATION']

        # Data distribution proportions for ['training', 'test', 'validation']
        data_dist = [0, 0, 0]
        data_dist[split] = 1
        dirs = os.listdir(datadir)
        removed_dirs = [dirname for dirname in dirs if not dirname.startswith(specific)]
        dirs_with_specific = [dirname for dirname in dirs if dirname.startswith(specific)]
        nb_variations = len(dirs_with_specific)
        
        # If specific name is specified
        if (specific):
            if removed_idx:
                removed_v_idx = removed_idx
            else:
                # Considering 80/0/20 division for train/test/validation
                nb_removed_v_idx = math.floor(0.2 * nb_variations)
                # Random.sample should not generate any duplicate indexes
                removed_v_idx = random.Random(SEED).sample(range(nb_variations), nb_removed_v_idx)

            removed_vars = []
            # If 'training', will use the non-removed function variations
            if split==0:
                removed_vars = removed_v_idx
            # If 'evaluation', will use the removed function variations
            elif split==2:
                # All of the rest function variations will not be considered
                removed_vars = [i for i in range(nb_variations) if i not in removed_v_idx]
                
            #print("The following variations will be excluded for the {} data set:".format(split_name[self.split]))
            #print(removed_vars)
            removed_dirs.extend([dirs_with_specific[i] for i in removed_vars])


        #print("The following directories will be excluded for the {} data set:".format(split_name[self.split]))
        #print(removed_dirs)
        # Read metric files
        self.read_data(datadir, removed_dirs, data_dist[0], data_dist[1], data_dist[2])



    def read_data(self, root_dir, removed_dirs, train_p=0, test_p=0, val_p=0):
        input_metrics = [
            "container_cpu_usage_seconds_total",
            "container_fs_writes_bytes_total",
            "container_memory_usage_bytes", 
            "container_network_transmit_packets_total"]
        control_metrics = [
            "container_network_receive_packets_total"
        ]
        self.ctrl_idxs = range(len(input_metrics), len(input_metrics) + len(control_metrics))

        # Start reading files
        input_metric_pool = {} # {'metric_1':[...], 'metric_2':[...], ..., 'metric_n':[...]}
        control_metric_pool = {}
        files = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            #if dirnames in functions:
            dirnames[:] = [d for d in dirnames if d not in removed_dirs]
            for filename in filenames:
                if filename.endswith('.json'):
                    filepath = os.path.join(dirpath, filename)
                    files.append(filepath)    

        # For metric in metric_names:
        for fname in files:
            # Read result metrics from datadir/metric.json 
            f = open(fname)
            data = json.load(f)
            results = data['data']['result']
            
            # Insert time-series into metric_pools
            metric = os.path.basename(fname).split('.')[0]
            truncate_at = (len(results[0]['values']) // self.seq_len) * self.seq_len
            # Input metrics
            if metric in input_metrics:
                # Check if we already have data of another experiment for this metric in the metric pool
                if metric in input_metric_pool:
                    values = input_metric_pool[metric]
                    # appending metrics of different experiments must be done carefuly as to avoid having the overlap of sequences
                    # for example, a sequence containing the end of an experiment and the beginning of another 
                    values.extend(results[0]['values'][:truncate_at]) 
                    input_metric_pool[metric] = values
                else:
                    input_metric_pool[metric] = results[0]['values'][:truncate_at]
            # Control metrics
            elif metric in control_metrics:
                if metric in control_metric_pool:
                    values = control_metric_pool[metric]
                    values.extend(results[0]['values'][:truncate_at]) 
                    control_metric_pool[metric] = values
                else:
                    control_metric_pool[metric] = results[0]['values'][:truncate_at]


        # Order metric pool by metric name
        input_metric_pool = dict(sorted(input_metric_pool.items()))
        control_metric_pool = dict(sorted(control_metric_pool.items()))
        
        # Join into one metric pool (assert that control metrics are at the end of pool)
        metric_pool = input_metric_pool | control_metric_pool
        
        # Assert that every metric has the same number of samples
        t_per_metric = [len(metric_pool[metric]) for metric in metric_pool.keys()]
        if sum(t_per_metric) != (len(t_per_metric) * t_per_metric[0]):
            raise Exception("Metric data does not have the same size across all metrics.")
        
        # Preprocess metric matrix
        self.metric_matrix = self.preprocess_data(metric_pool, t_per_metric, train_p, test_p, val_p)


    def preprocess_data(self, metric_pool, t_per_metric, train_p=0, test_p=0, val_p=0):
        self.metric_names = metric_pool.keys()

        # Compute sequence grouping
        if (self.shuffle):
            self.compute_seqs_shuffle(t_per_metric=t_per_metric, train_p=train_p, test_p=test_p, val_p=val_p)
        else:
            self.compute_seqs_ordered(t_per_metric=t_per_metric, train_p=train_p, test_p=test_p, val_p=val_p)

        # Normalize metric values
        np_metrics = np.empty((len(metric_pool.items()), t_per_metric[0]), dtype=np.float32)
        for i,metric in enumerate(metric_pool.items()):
            values = np.array(metric[1], dtype=np.float32)[:,1]
            np_metrics[i] = values

        self.scaler = preprocessing.MaxAbsScaler()
        scaled_metrics = self.scaler.fit_transform(np_metrics.transpose())

        return scaled_metrics


    def compute_seqs_ordered(self, t_per_metric, train_p, test_p, val_p):
        # Map observations into sequences of samples
        self.valid_seq_list = []
        n_seq = t_per_metric[0] // self.seq_len

        offset = 0
        if self.split==0:
            n_seq = math.floor(n_seq*train_p)
        elif self.split==1:
            offset = math.floor(n_seq*train_p)
            n_seq = math.floor(n_seq*test_p)   
        else:
            offset = math.floor(n_seq*(train_p+test_p))
            n_seq = math.floor(n_seq*val_p)  

        for i in range(offset, n_seq+offset):
            start = i * self.seq_len
            end = start + self.seq_len - 1
            self.valid_seq_list.append((start, end))


    def compute_seqs_shuffle(self, t_per_metric, train_p, test_p, val_p):
        # Map observations into sequences of samples
        self.valid_seq_list = []
        valid_seq_list_aux = []
        n_seq = t_per_metric[0] // self.seq_len

        for i in range(n_seq):
            start = i * self.seq_len
            end = start + self.seq_len - 1
            valid_seq_list_aux.append((start, end))

        random.Random(SEED).shuffle(valid_seq_list_aux)

        offset = 0
        if self.split==0:
            n_seq = math.floor(n_seq*train_p)
        elif self.split==1:
            offset = math.floor(n_seq*train_p)
            n_seq = math.floor(n_seq*test_p)   
        else:
            offset = math.floor(n_seq*(train_p+test_p))
            n_seq = math.floor(n_seq*val_p)  

        self.valid_seq_list = valid_seq_list_aux[offset:n_seq+offset]

    def __metric_names__(self):
        return list(self.metric_names)

    def __ctrl_idxs__(self):
        return self.ctrl_idxs

    def __seq_len__(self):
        return self.seq_len

    def __len__(self):
        """
        Return the total number of samples (sequences).
        """
        return len(self.valid_seq_list)


    def __getitem__(self, index):
        """
        Returns a np.ndarray with shape (#L, #M) containing the sequence of metrics at index. 
        Where #L is the sequence's length and #M is the number of metrics.
        """
        seq_start, seq_end = self.valid_seq_list[index]
        # seq_end+1 because the slice operator ':' does not include the end
        sample_seq = self.metric_matrix[seq_start:seq_end+1, :]

        return sample_seq