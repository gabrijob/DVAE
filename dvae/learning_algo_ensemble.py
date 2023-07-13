#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import socket
import datetime
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from .utils import myconf, get_logger, loss_KLD, loss_PIQD
from .dataset import ensemble_dataset
from .model import build_SRNN, build_cSRNN


class LearningAlgorithm_ensemble():

    """
    Basical class for curr_model building, including:
    - read common paramters for different models
    - define data loader
    - define loss function as a class member
    """

    def __init__(self, params):
        # Load config parser
        self.params = params
        self.config_file = self.params['cfg']
        if not os.path.isfile(self.config_file):
            raise ValueError('Invalid config file path')    
        self.cfg = myconf()
        self.cfg.read(self.config_file)
        self.model_name = self.cfg.get('Network', 'name')
        self.dataset_name = self.cfg.get('DataFrame', 'dataset_name')

        # Get host name and date
        self.hostname = socket.gethostname()
        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M")
        
        # Load curr_model parameters
        self.use_cuda = self.cfg.getboolean('Training', 'use_cuda')
        self.device = 'cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu'


    def build_model(self):
        if self.model_name == 'SRNN':
            self.curr_model = build_SRNN(cfg=self.cfg, device=self.device)
        elif self.model_name == 'cSRNN':
            self.curr_model = build_cSRNN(cfg=self.cfg, device=self.device)
        else:
            print('Error: wrong curr_model type')
        

    def init_optimizer(self):
        optimization  = self.cfg.get('Training', 'optimization')
        lr = self.cfg.getfloat('Training', 'lr')
        if optimization == 'adam': # could be extend to other optimizers
            optimizer = torch.optim.Adam(self.curr_model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(self.curr_model.parameters(), lr=lr)
        return optimizer


    def get_basic_info(self):
        basic_info = []
        basic_info.append('HOSTNAME: ' + self.hostname)
        basic_info.append('Time: ' + self.date)
        basic_info.append('Device for training: ' + self.device)
        if self.device == 'cuda':
            basic_info.append('Cuda verion: {}'.format(torch.version.cuda))
        basic_info.append('Model name: {}'.format(self.model_name))
        basic_info.append('Total params: %.2fM' % (sum(p.numel() for p in self.curr_model.parameters()) / 1000000.0))
        
        return basic_info


    def train_model(self, specific):
        ############
        ### Init ###
        ############

        # Build curr_model
        self.build_model()

        # Set module.training = True
        self.curr_model.train()
        torch.autograd.set_detect_anomaly(True)

        # Create directory for results
        if not self.params['reload']:
            saved_root = self.cfg.get('User', 'saved_root')
            z_dim = self.cfg.getint('Network','z_dim')
            tag = self.cfg.get('Network', 'tag')
            filename = "{}_{}_{}_z_dim={}_ensemble".format(self.dataset_name, self.date, tag, z_dim)
            save_dir = os.path.join(saved_root, filename, specific)
            if not(os.path.isdir(save_dir)):
                os.makedirs(save_dir)
        else:
            tag = self.cfg.get('Network', 'tag')
            save_dir = self.params['model_dir']
            

        # Save the curr_model configuration
        save_cfg = os.path.join(save_dir, 'config.ini')
        shutil.copy(self.config_file, save_cfg)

        # Create logger
        log_file = os.path.join(save_dir, 'log.txt')
        logger_type = self.cfg.getint('User', 'logger_type')
        logger = get_logger(log_file, logger_type)

        # Print basical infomation
        for log in self.get_basic_info():
            logger.info(log)
        logger.info('In this experiment, result will be saved in: ' + save_dir)

        # Print curr_model infomation (optional)
        if self.cfg.getboolean('User', 'print_model'):
            for log in self.curr_model.get_info():
                logger.info(log)

        # Init optimizer
        optimizer = self.init_optimizer()

        # Create data loader
        if self.dataset_name == 'METRIC': # Only metric dataset for now
            train_dataloader, val_dataloader, train_num, val_num = ensemble_dataset.build_dataloader(self.cfg, specific)
        else:
            logger.error('Unknown datset')
        logger.info('Training samples: {}'.format(train_num))
        logger.info('Validation samples: {}'.format(val_num))
        
        ######################
        ### Batch Training ###
        ######################

        # Load training parameters
        epochs = self.cfg.getint('Training', 'epochs')
        early_stop_patience = self.cfg.getint('Training', 'early_stop_patience')
        save_frequency = self.cfg.getint('Training', 'save_frequency')
        beta = self.cfg.getfloat('Training', 'beta')
        gamma = 0.1
        alpha = 0.1
        kl_warm = 0
        qd_warm = 0

        # Create python list for loss
        if not self.params['reload']:
            train_loss = np.zeros((epochs,))
            val_loss = np.zeros((epochs,))
            train_recon = np.zeros((epochs,))
            train_kl = np.zeros((epochs,))
            train_qd = np.zeros((epochs,))
            val_recon = np.zeros((epochs,))
            val_kl = np.zeros((epochs,))
            val_qd = np.zeros((epochs,))
            best_val_loss = np.inf
            cpt_patience = 0
            cur_best_epoch = 0 #epochs
            best_state_dict = self.curr_model.state_dict()
            best_optim_dict = optimizer.state_dict()
            start_epoch = -1
        else:
            cp_file = os.path.join(save_dir, '{}_checkpoint.pt'.format(self.model_name))
            checkpoint = torch.load(cp_file)
            self.curr_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            start_epoch = checkpoint['epoch']
            loss_log = checkpoint['loss_log']
            logger.info('Resuming trainning: epoch: {}'.format(start_epoch))
            train_loss = np.pad(loss_log['train_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            val_loss = np.pad(loss_log['val_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            train_recon = np.pad(loss_log['train_recon'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            train_kl = np.pad(loss_log['train_kl'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            train_qd = np.pad(loss_log['train_qd'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            val_recon = np.pad(loss_log['val_recon'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            val_kl = np.pad(loss_log['val_kl'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            val_qd = np.pad(loss_log['val_qd'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            best_val_loss = checkpoint['best_val_loss']
            cpt_patience = 0
            cur_best_epoch = start_epoch
            best_state_dict = self.curr_model.state_dict()
            best_optim_dict = optimizer.state_dict()
            

        # Train with mini-batch SGD
        for epoch in range(start_epoch+1, epochs):
            
            start_time = datetime.datetime.now()

            # KL warm-up
            if epoch % 10 == 0 and kl_warm < 1:
                #kl_warm = (epoch // 10) * 0.2 
                kl_warm = 1
                logger.info('KL warm-up, anneal coeff: {}'.format(kl_warm))
            
            # QD warm-up
            if epoch >= 30 and qd_warm < 1:
                qd_warm = 1.0
                logger.info('QD warm-up, anneal coeff: {}'.format(qd_warm))


            # Batch training
            for _, batch_data in enumerate(train_dataloader):
                batch_data = batch_data.to(self.device)
                
                if self.dataset_name == 'METRIC': # Only metric dataset for now
                    # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
                    batch_data = batch_data.permute(1, 0, 2)
                    recon_batch_data = self.curr_model(batch_data)
                    # Get only x input from batch data
                    xcut_batch_data = batch_data[:, :, :self.curr_model.x_dim]
                    
                    loss_fn = torch.nn.MSELoss(reduction='sum')
                    loss_recon = loss_fn(xcut_batch_data, recon_batch_data)
                else:
                    logger.error('Unknown datset')

                seq_len, bs, _ = self.curr_model.z_mean.shape
                loss_recon = loss_recon / (seq_len * bs)

                # Kullback-Lieber Loss
                loss_kl = loss_KLD(self.curr_model.z_mean, self.curr_model.z_logvar, self.curr_model.z_mean_p, self.curr_model.z_logvar_p)
                loss_kl = kl_warm * beta * loss_kl / (seq_len * bs)

                # Quality-Driven Prediction Interval Loss
                loss_qd, _, _ = loss_PIQD(xcut_batch_data, self.curr_model.y_lower_bound, self.curr_model.y_upper_bound, alpha=alpha)
                loss_qd = loss_qd * gamma * qd_warm

                loss_tot = loss_recon + loss_kl + loss_qd
                optimizer.zero_grad()
                loss_tot.backward()
                optimizer.step()

                train_loss[epoch] += loss_tot.item() * bs
                train_recon[epoch] += loss_recon.item() * bs
                train_kl[epoch] += loss_kl.item() * bs
                train_qd[epoch] += loss_qd.item() * bs

            # Validation
            for _, batch_data in enumerate(val_dataloader):

                batch_data = batch_data.to(self.device)

                if self.dataset_name == 'METRIC':
                    # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
                    batch_data = batch_data.permute(1, 0, 2)
                    recon_batch_data = self.curr_model(batch_data)
                    # Get only x input from batch data
                    xcut_batch_data = batch_data[:, :, :self.curr_model.x_dim]

                    loss_fn = torch.nn.MSELoss(reduction='sum')
                    loss_recon = loss_fn(xcut_batch_data, recon_batch_data)
                else:
                    logger.error('Unknown datset')

                seq_len, bs, _ = self.curr_model.z_mean.shape
                loss_recon = loss_recon / (seq_len * bs)
                
                # Kullback-Lieber Loss
                loss_kl = loss_KLD(self.curr_model.z_mean, self.curr_model.z_logvar, self.curr_model.z_mean_p, self.curr_model.z_logvar_p)
                loss_kl = kl_warm * beta * loss_kl / (seq_len * bs)

                # Quality-Driven Prediction Interval Loss
                loss_qd, _, _ = loss_PIQD(xcut_batch_data, self.curr_model.y_lower_bound, self.curr_model.y_upper_bound, alpha=alpha)
                loss_qd = loss_qd * gamma * qd_warm

                loss_tot = loss_recon + loss_kl + loss_qd

                val_loss[epoch] += loss_tot.item() * bs
                val_recon[epoch] += loss_recon.item() * bs
                val_kl[epoch] += loss_kl.item() * bs
                val_qd[epoch] += loss_qd.item() * bs

            # Loss normalization
            train_loss[epoch] = train_loss[epoch]/ train_num
            val_loss[epoch] = val_loss[epoch] / val_num
            train_recon[epoch] = train_recon[epoch] / train_num 
            train_kl[epoch] = train_kl[epoch]/ train_num
            train_qd[epoch] = train_qd[epoch] / train_num
            val_recon[epoch] = val_recon[epoch] / val_num 
            val_kl[epoch] = val_kl[epoch] / val_num
            val_qd[epoch] = val_qd[epoch] / val_num
            
            # Early stop patiance
            if val_loss[epoch] < best_val_loss and qd_warm>=1: #or kl_warm <1 :
                best_val_loss = val_loss[epoch]
                cpt_patience = 0
                best_state_dict = self.curr_model.state_dict()
                best_optim_dict = optimizer.state_dict()
                cur_best_epoch = epoch
            else:
                cpt_patience += 1

            # Training time
            end_time = datetime.datetime.now()
            interval = (end_time - start_time).seconds / 60
            logger.info('Epoch: {} training time {:.2f}m'.format(epoch, interval))
            logger.info('Train => tot: {:.2f} recon {:.2f} KL {:.2f} QD {:.2f} Val => tot: {:.2f} recon {:.2f} KL {:.2f} QD {:.2f}'.format(train_loss[epoch], train_recon[epoch], train_kl[epoch], train_qd[epoch], val_loss[epoch], val_recon[epoch], val_kl[epoch], val_qd[epoch]))


            # Stop traning if early-stop triggers
            if cpt_patience == early_stop_patience and kl_warm >= 1.0:
                logger.info('Early stop patience achieved')
                break

            # Save curr_model parameters regularly
            if epoch % save_frequency == 0:
                loss_log = {'train_loss': train_loss[:cur_best_epoch+1],
                            'val_loss': val_loss[:cur_best_epoch+1],
                            'train_recon': train_recon[:cur_best_epoch+1],
                            'train_kl': train_kl[:cur_best_epoch+1], 
                            'train_qd': train_qd[:cur_best_epoch+1], 
                            'val_recon': val_recon[:cur_best_epoch+1], 
                            'val_kl': val_kl[:cur_best_epoch+1],
                            'val_qd': val_qd[:cur_best_epoch+1]}
                save_file = os.path.join(save_dir, self.model_name + '_checkpoint.pt')
                torch.save({'epoch': cur_best_epoch,
                            'best_val_loss': best_val_loss,
                            'cpt_patience': cpt_patience,
                            'model_state_dict': best_state_dict,
                            'optim_state_dict': best_optim_dict,
                            'loss_log': loss_log
                        }, save_file)
                logger.info('Epoch: {} ===> checkpoint stored with current best epoch: {}'.format(epoch, cur_best_epoch))

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        
        # Save the final weights of network with the best validation loss
        save_file = os.path.join(save_dir, self.model_name + '_final_epoch.pt')
        torch.save(best_state_dict, save_file)
        
        # Save the training loss and validation loss
        train_loss = train_loss[:epoch+1]
        val_loss = val_loss[:epoch+1]
        train_recon = train_recon[:epoch+1]
        train_kl = train_kl[:epoch+1]
        train_qd = train_qd[:epoch+1]
        val_recon = val_recon[:epoch+1]
        val_kl = val_kl[:epoch+1]
        val_qd = val_qd[:epoch+1]
        loss_file = os.path.join(save_dir, 'loss_model.pckl')
        with open(loss_file, 'wb') as f:
            pickle.dump([train_loss, val_loss, train_recon, train_kl, train_qd, val_recon, val_kl, val_qd], f)

        # Save the loss figure
        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_loss, label='training loss')
        plt.plot(val_loss, label='validation loss')
        plt.legend(fontsize=16, title=self.model_name, title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_{}.png'.format(tag))
        plt.savefig(fig_file)
        plt.close()

        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_recon, label='Training')
        plt.plot(val_recon, label='Validation')
        plt.legend(fontsize=16, title='{}: Recon. Loss'.format(self.model_name), title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_recon_{}.png'.format(tag))
        plt.savefig(fig_file) 
        plt.close()


        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_kl, label='Training')
        plt.plot(val_kl, label='Validation')
        plt.legend(fontsize=16, title='{}: KL Divergence'.format(self.model_name), title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_KLD_{}.png'.format(tag))
        plt.savefig(fig_file)
        plt.close()

        
        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_qd, label='Training')
        plt.plot(val_qd, label='Validation')
        plt.legend(fontsize=16, title='{}: Quality Driven Prediction Interval Loss'.format(self.model_name), title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_QD_{}.png'.format(tag))
        plt.savefig(fig_file)
        plt.close()
   
                
    def train_ensemble(self):
        
        data_dir = self.cfg.get('User', 'data_dir')
        
        # For each function, train a curr_model 
        functions = ["abssin", "abscos", "sin", "cos", "bell", "linear", "log"]
        for f in functions:        
            print("Training curr_model for the " + f + " function.")
            self.train_model(f)

    def train(self):
        self.train_ensemble()