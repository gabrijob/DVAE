#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt
"""
import torch

def loss_ISD(x, y):
    y = y + 1e-10
    ret = torch.sum( x/y - torch.log(x/y) - 1)
    return ret

def loss_KLD(z_mean, z_logvar, z_mean_p=0, z_logvar_p=0):
    ret = -0.5 * torch.sum(z_logvar - z_logvar_p 
                - torch.div(z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp()+1e-10))
    return ret

def loss_JointNorm(x, y, nfeats=3):
    seq_len, bs, _ = x.shape
    x = x.reshape(seq_len, bs, -1, nfeats)
    y = y.reshape(seq_len, bs, -1, nfeats)
    ret = torch.sum(torch.norm(x-y, dim=-1))
    return ret

def loss_MPJPE(x, y, nfeats=3):
    seq_len, bs, _ = x.shape
    x = x.reshape(seq_len, bs, -1, nfeats)
    y = y.reshape(seq_len, bs, -1, nfeats)
    ret = (x-y).norm(dim=-1).mean(dim=-1).sum()
    return ret

def loss_MSE(x, y):
    ret = torch.mean((x-y)**2)
    return ret

def loss_PIQD(y_true, y_lower, y_upper, alpha, soft_factor=150, lagrangian=0.1):
    seq_len, batch_size, _ = y_true.shape

    k_hu = torch.max(torch.zeros(y_true.shape), torch.sign(y_upper - y_true))
    k_hl = torch.max(torch.zeros(y_true.shape), torch.sign(y_true - y_lower))
    k_hard = k_hu * k_hl

    k_su = torch.sigmoid((y_upper - y_true) * soft_factor)
    k_sl = torch.sigmoid((y_true - y_lower) * soft_factor)
    k_soft = k_su * k_sl

    MPIW_c = torch.sum((y_upper - y_lower) * k_hard) / torch.sum(k_hard)
    PICP_soft = torch.mean(k_soft)

    loss = MPIW_c + lagrangian * (seq_len*batch_size) / (alpha*(1-alpha)) * torch.max(torch.zeros((1)), (1-alpha) - PICP_soft)**2
    return loss, MPIW_c, PICP_soft


# def loss_ISD(x, y):
#     seq_len, bs, _ = x.shape
#     ret = torch.sum( x/y - torch.log(x/y) - 1)
#     ret = ret / (bs * seq_len)
#     return ret

# def loss_KLD(z_mean, z_logvar, z_mean_p=0, z_logvar_p=0):
#     if len(z_mean.shape) == 3:
#         seq_len, bs, _ = z_mean.shape
#     elif len(z_mean.shape) == 2:
#         seq_len = 1
#         bs, _ = z_mean.shape
#     ret = -0.5 * torch.sum(z_logvar - z_logvar_p 
#                 - torch.div(z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp()))
#     ret = ret / (bs * seq_len)
#     return ret

# def loss_JointNorm(x, y, nfeats=3):
#     seq_len, bs, _ = x.shape
#     x = x.reshape(seq_len, bs, -1, nfeats)
#     y = y.reshape(seq_len, bs, -1, nfeats)
#     return torch.mean(torch.norm(x-y, dim=-1))



