import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random


class RealSHaze:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        self.listfile = sorted(os.listdir(config.data.data_dir))

        self.trainlist = ['haze']
        self.testlist = ['haze']


    def get_loaders(self, parse_patches=True, validation='haze'):
        train_dataset = RealSHazeDataset(dir=os.path.join(self.config.data.data_dir),
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        filelist=self.trainlist,
                                        istrain = True,
                                        parse_patches=parse_patches)


        val_dataset = RealSHazeDataset(dir=os.path.join(self.config.data.data_dir),
                                      n=self.config.training.patch_n,
                                      patch_size=self.config.data.image_size,
                                      transforms=self.transforms,
                                      filelist=self.testlist,
                                      istrain = False,
                                      parse_patches=parse_patches)



        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size ,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)
        return train_loader, val_loader


class RealSHazeDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, filelist=None, istrain=True, clip_length=3, parse_patches=True):
        super().__init__()
        print('-dir-', dir)
        self.istrain = istrain
        self.clip_length = clip_length
        self.dir = dir
        train_list = []
        input_names = []
        gt_names = []
        A_names = []
        for i in range(len(filelist)):
            id = os.path.join(self.dir, filelist[i])
            inpdir = os.path.join(id)
            gtdir = os.path.join(id)
            listinpdir = sorted(os.listdir(inpdir))
            for j in range(len(listinpdir)):
                input_names.append(os.path.join(inpdir, listinpdir[j]))
            listgtdir = sorted(os.listdir(gtdir))
            for j in range(len(listgtdir)):
                gt_names.append(os.path.join(gtdir, listgtdir[j]))
    

        print('len(input_names),len(gt_names) = ', len(input_names), len(gt_names), len(A_names))
        print(input_names[0], gt_names[0])
        print(input_names[-1], gt_names[-1])
        


        self.input_names = input_names
        self.gt_names = gt_names


        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches
        # print('-self.patch_size self.n -',self.patch_size, self.n)


    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
   
        datasetname = re.split('/', input_name)[-3]
        img_vid = re.split('/', input_name)[-2]
        img_id = re.split('/', input_name)[-1][:-4]
        img_id = datasetname + '__' + img_vid + '__' + img_id




        input_img = PIL.Image.open(input_name)
        gt_img = PIL.Image.open(gt_name)

        # wd_new, ht_new = input_img.size
        wd_new = 512
        ht_new = 512
        input_img = input_img.resize((wd_new, ht_new), PIL.Image.NEAREST)
        gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.NEAREST)

        return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
