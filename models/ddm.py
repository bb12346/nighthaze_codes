import os
import time
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils
import cv2


from models.models_mae import mae_vit_large_patch16_dec512d8b

from torchvision.transforms.functional import crop
import random
import torch.optim as optim
import torch.nn.functional as F






def data_transform(X):
    imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).cuda()
    imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).cuda()
    # print('-imagenet_mean-',imagenet_mean.shape)
    imagenet_mean = imagenet_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    imagenet_std = imagenet_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    # print('-imagenet_mean-',imagenet_mean.shape)
    X = X - imagenet_mean
    X = X / imagenet_std
    return X


def inverse_data_transform(X):
    imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).cuda()
    imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).cuda()
    # print('-imagenet_mean-',imagenet_mean.shape)
    imagenet_mean = imagenet_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    imagenet_std = imagenet_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    return torch.clip((X * imagenet_std + imagenet_mean), 0, 1)


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

class EMAHelper(object):
    def __init__(self, mu=0.9995):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict






class NightHaze(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.Dehaze = mae_vit_large_patch16_dec512d8b()
        self.Dehaze.to(self.device)



        self.Dehaze = torch.nn.DataParallel(self.Dehaze)


        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.Dehaze.parameters()), lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                        betas=(0.9, 0.999), amsgrad=self.config.optim.amsgrad, eps=self.config.optim.eps)
           

        self.start_epoch, self.step = 0, 0



    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.Dehaze.load_state_dict(checkpoint['Dehaze'], strict=True)
        # self.ET.load_state_dict(checkpoint['ET_state_dict'], strict=True)
        # self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))




    def aug(self,inp,A_inp):

        return inp,A_inp





    def train(self, DATASET):
        cudnn.benchmark = True
        shuffle = False
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)


        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0

                

