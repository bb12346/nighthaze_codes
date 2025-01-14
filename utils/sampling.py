import torch
import utils.logging
import os
import torchvision
from torchvision.transforms.functional import crop
import random


# This script is adapted from the following repository: https://github.com/ermongroup/ddim

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def generalized_steps(x_cond, seq, model):
    with torch.no_grad():
        n = x_cond.size(0)
        # t = torch.ones_like(x_cond[:,0,0,0]).cuda()
        # # print('t',t.shape)
        # for i in range(len(t)):
        #     t[i] = 0.20
        et = model(x_cond)

    return et, None


def generalized_steps_overlapping(x_cond, seq, model, corners=None, p_size=None, manual_batching=True):
    with torch.no_grad():
        n = x_cond.size(0)

        # t = torch.ones_like(x_cond[:,0,0,0]).cuda()
        # # print('t',t.shape)
        # t[0] = 0.20
        # # seq_next = [-1] + list(seq[:-1])
        # # x0_preds = []
        # xs = [x]
        et_output = torch.zeros_like(x_cond, device=x_cond.device)

        x_grid_mask = torch.zeros_like(x_cond, device=x_cond.device)
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1

        if manual_batching:
            manual_batching_size = 64
            x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in corners], dim=0)
            for i in range(0, len(corners), manual_batching_size):
                outputs = model(x_cond_patch[i:i+manual_batching_size])
                for idx, (hi, wi) in enumerate(corners[i:i+manual_batching_size]):
                    et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]
        else:
            for (hi, wi) in corners:
                x_cond_patch = crop(x_cond, hi, wi, p_size, p_size)
                x_cond_patch = data_transform(x_cond_patch)
                et_output[:, :, hi:hi + p_size, wi:wi + p_size] += model(x_cond_patch)

        et = torch.div(et_output, x_grid_mask)

    return et, None
