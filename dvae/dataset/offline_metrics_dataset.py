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

# Get JSON (or other) files with the results of metrics queries from several days/workload sessions
# Online training will come from http requesting at runtime (later)
def build_dataloader(cfg):
    
    # Load config params
    data_dir = cfg.get('User', 'data_dir')
    shuffle = cfg.getboolean('DataFrame', 'shuffle')
    batch_size = cfg.getint('DataFrame', 'batch_size')
    num_workers = cfg.getint('DataFrame', 'num_workers')
    sequence_len = cfg.getint('DataFrame', 'sequence_len')

    # Training dataset
    split = 0 # train
    train_dataset = OfflinePrometheusMetrics(svc_datadir=data_dir, seq_len=sequence_len, shuffle=shuffle, split=split)
    split = 2 # validation
    val_dataset = OfflinePrometheusMetrics(svc_datadir=data_dir, seq_len=sequence_len, shuffle=shuffle, split=split)
    train_num = train_dataset.__len__()
    val_num = val_dataset.__len__()
    
    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                shuffle=shuffle, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                shuffle=shuffle, num_workers=num_workers)

    return train_dataloader, val_dataloader, train_num, val_num

    

class OfflinePrometheusMetrics(Dataset):
    def __init__(self, svc_datadir, seq_len, shuffle, split=0):
        self.seq_len = seq_len
        self.shuffle = shuffle

        # {'metric_1':[...], 'metric_2':[...], ..., 'metric_n':[...]}
        self.metric_pool = {}
        
        dirs = os.listdir(svc_datadir)
        files = []
        for dir in dirs:
            f_in_dir = os.listdir(svc_datadir+'/'+dir)
            files.extend([dir+'/'+f for f in f_in_dir if os.path.isfile(svc_datadir+'/'+dir+'/'+f)])
        #print(*files, sep="\n")
        
        # For metric in metric_names:
        for fname in files:
            
            # Read result metrics from datadir/metric.json 
            f = open(svc_datadir+'/'+fname)
            data = json.load(f)
            results = data['data']['result']
            
            # Insert time-series into metric_pool
            #metric = results[0]['metric']['__name__']
            metric = os.path.basename(fname).split('.')[0]
            #print(metric)
            truncate_at = (len(results[0]['values']) // self.seq_len) * self.seq_len
            if metric in self.metric_pool:
                values = self.metric_pool[metric]
                # appending metrics of different experiments must be done carefuly as to avoid having the overlap of sequences
                # for example, a sequence containing the end of an experiment and the beginning of another 
                values.extend(results[0]['values'][:truncate_at]) 
                self.metric_pool[metric] = values
            else:
                self.metric_pool[metric] = results[0]['values'][:truncate_at]

        # Assert that every metric has the same number of samples
        t_per_metric = [len(self.metric_pool[metric]) for metric in self.metric_pool.keys()]
        if sum(t_per_metric) != (len(t_per_metric) * t_per_metric[0]):
            raise Exception("Metric data does not have the same size across all metrics.")
        
        # Preprocess metric matrix
        self.metric_matrix = self.preprocess_data(split, t_per_metric)

        
    
    def preprocess_data(self, split, t_per_metric):
        # Order metric pool by metric name
        self.metric_pool = dict(sorted(self.metric_pool.items()))

        # Compute sequence grouping
        self.compute_seqs(split=split, t_per_metric=t_per_metric)

        # Normalize metric values
        np_metrics = np.empty((len(self.metric_pool.items()), t_per_metric[0]), dtype=np.float32)
        for i,metric in enumerate(self.metric_pool.items()):
            values = np.array(metric[1], dtype=np.float32)[:,1]
            np_metrics[i] = values

        self.scaler = preprocessing.MaxAbsScaler()
        scaled_metrics = self.scaler.fit_transform(np_metrics.transpose())

        return scaled_metrics


    def compute_seqs(self, split, t_per_metric, train_p=0.7, test_p=0., val_p=0.3):
        # Map observations into sequences of samples
        self.valid_seq_list = []
        n_seq = t_per_metric[0] // self.seq_len

        offset = 0
        if split==0:
            n_seq = math.floor(n_seq*train_p)
        elif split==1:
            offset = math.floor(n_seq*train_p)
            n_seq = math.floor(n_seq*test_p)   
        else:
            offset = math.floor(n_seq*(train_p+test_p))
            #n_seq -= offset
            n_seq = math.floor(n_seq*val_p)  

        for i in range(offset, n_seq+offset):
            start = i * self.seq_len
            end = start + self.seq_len - 1
            self.valid_seq_list.append((start, end))

        if self.shuffle:
            random.shuffle(self.valid_seq_list)


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

        #sample_seq = np.empty((self.seq_len, len(self.metric_pool.items())), dtype=np.float32)

        # Variable metric in the shape of ('name', [(timestamp, value), ...])
        #for metric in self.metric_pool.items():
        #    name = metric[0]
        #    sequence = np.array(metric[1][1][seq_start:seq_end]) 
        #    sample[name] = sequence

        #for i in range(seq_start, seq_end):
        #    sample_i = []
        #    for metric in self.metric_pool.items():
        #        value = float(metric[1][i][1])
        #        if (np.isnan(value)): # Check for pos-inf, neg-inf, nan
        #            sample_i.append(0)
        #        else:
        #            sample_i.append(value)
        #    sample_seq[i-seq_start] = np.array(sample_i, dtype=np.float32)
            #print(sample_seq)
            
            #if ~np.isfinite(np.sum(sample_seq[i-seq_start])):
            #    print(sample_seq)
            #    print("At " + str(seq_start) + "," + str(seq_end)) 

        return sample_seq