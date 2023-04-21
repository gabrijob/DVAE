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
SEED = 3


# Get JSON (or other) files with the results of metrics queries from several days/workload sessions
# Online training will come from http requesting at runtime (later)
def build_dataloader(cfg):
    
    # Load config params
    data_dir = cfg.get('User', 'data_dir')
    shuffle = cfg.getboolean('DataFrame', 'shuffle')
    batch_size = cfg.getint('DataFrame', 'batch_size')
    num_workers = cfg.getint('DataFrame', 'num_workers')
    sequence_len = cfg.getint('DataFrame', 'sequence_len')
    is_functions = cfg.getboolean('DataFrame', 'functions')

    # Training dataset
    split = 0 # train
    train_dataset = OfflinePrometheusMetrics(svc_datadir=data_dir, seq_len=sequence_len, shuffle=shuffle, split=split, is_functions=is_functions)
    split = 2 # validation
    val_dataset = OfflinePrometheusMetrics(svc_datadir=data_dir, seq_len=sequence_len, shuffle=shuffle, split=split, is_functions=is_functions)
    train_num = train_dataset.__len__()
    val_num = val_dataset.__len__()
    
    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                shuffle=shuffle, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                shuffle=shuffle, num_workers=num_workers)

    return train_dataloader, val_dataloader, train_num, val_num

    

class OfflinePrometheusMetrics(Dataset):
    def __init__(self, svc_datadir, seq_len, shuffle, split=0, is_functions=False):
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.split = split

        functions = ["abssin", "abscos", "sin", "cos", "bell", "linear", "log"]
        
        if (is_functions):
            # Random.sample should not generate any duplicate indexes
            #removed_idx = random.Random(SEED).sample(range(0, len(functions)-1), 2)
            removed_idx = [6,0]

            # Data distribution proportions for ['training', 'test', 'validation']
            data_dist = [0, 0, 0]
            data_dist[split] = 1

            # If 'training', use the non-removed functions
            if split==0:
                del functions[removed_idx[0]]
                del functions[removed_idx[1]]
            # If 'test' or 'evaluation', use only one of the removed functions
            else:
                functions = [functions.pop(removed_idx[split-1])]

            print(functions)
        else:
            # Data distribution proportions for ['training', 'test', 'validation']
            data_dist = [0.6, 0.2, 0.2]

        self.read_data(svc_datadir, functions, data_dist[0], data_dist[1], data_dist[2])


    def read_data(self, svc_datadir, functions, train_p=0, test_p=0, val_p=0):
        
        # Start reading files
        metric_pool = {} # {'metric_1':[...], 'metric_2':[...], ..., 'metric_n':[...]}
        files = []
        for func_dir in functions:
            exp_dirs = os.listdir(svc_datadir+'/'+func_dir)
            for exp_dir in exp_dirs:
                f_in_dir = os.listdir(svc_datadir+'/'+func_dir+'/'+exp_dir)
                files.extend([func_dir+'/'+exp_dir+'/'+f for f in f_in_dir if os.path.isfile(svc_datadir+'/'+func_dir+'/'+exp_dir+'/'+f)])

        # For metric in metric_names:
        for fname in files:
            # Read result metrics from datadir/metric.json 
            f = open(svc_datadir+'/'+fname)
            data = json.load(f)
            results = data['data']['result']
            
            # Insert time-series into metric_pool
            metric = os.path.basename(fname).split('.')[0]
            truncate_at = (len(results[0]['values']) // self.seq_len) * self.seq_len
            if metric in metric_pool:
                values = metric_pool[metric]
                # appending metrics of different experiments must be done carefuly as to avoid having the overlap of sequences
                # for example, a sequence containing the end of an experiment and the beginning of another 
                values.extend(results[0]['values'][:truncate_at]) 
                metric_pool[metric] = values
            else:
                metric_pool[metric] = results[0]['values'][:truncate_at]


        # Assert that every metric has the same number of samples
        t_per_metric = [len(metric_pool[metric]) for metric in metric_pool.keys()]
        if sum(t_per_metric) != (len(t_per_metric) * t_per_metric[0]):
            raise Exception("Metric data does not have the same size across all metrics.")
        
        # Preprocess metric matrix
        self.metric_matrix = self.preprocess_data(metric_pool, t_per_metric, train_p, test_p, val_p)


    def preprocess_data(self, metric_pool, t_per_metric, train_p=0, test_p=0, val_p=0):
        # Order metric pool by metric name
        metric_pool = dict(sorted(metric_pool.items()))
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
        sample_seq = self.metric_matrix[seq_start:seq_end, :]

        return sample_seq