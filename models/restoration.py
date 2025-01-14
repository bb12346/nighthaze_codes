import torch
import torch.nn as nn
import utils
import torchvision
import numpy as np
import cv2


import os
from torchvision.transforms.functional import crop

def data_transform(X):
    imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).cpu()
    imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).cpu()
    # print('-imagenet_mean-',imagenet_mean.shape)
    imagenet_mean = imagenet_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    imagenet_std = imagenet_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    # print('-imagenet_mean-',imagenet_mean.shape)
    X = X - imagenet_mean
    X = X / imagenet_std
    return X


def inverse_data_transform(X):
    imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).cpu()
    imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).cpu()
    # print('-imagenet_mean-',imagenet_mean.shape)
    imagenet_mean = imagenet_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    imagenet_std = imagenet_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    return torch.clip((X * imagenet_std + imagenet_mean), 0, 1)


class NightHazeRestoration:
    def __init__(self, nighthaze, args, config):
        super(NightHazeRestoration, self).__init__()
        self.args = args
        self.config = config
        self.nighthaze = nighthaze

        print("-args.resume-",args.resume)
        if os.path.isfile(args.resume):
            self.nighthaze.load_ddm_ckpt(args.resume, ema=True)
            self.nighthaze.Dehaze.eval()
            # self.diffusion.teachermodel.eval()
            # self.diffusion.studentmodel.eval()
        else:
            print('Pre-trained nighthaze model path is missing!')

    def restore(self, val_loader, validation='haze', sid = None):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset, validation)
        with torch.no_grad():
            # count = 132
            for i, (x, y) in enumerate(val_loader):
                print('-restore-',x.shape,y)
                # if i < 230:
                    # continue
                if sid:
                    if sid+'__' in y[0]:

                        print(self.args.image_folder, self.config.data.dataset)
                        print(i, x.shape, y)
 

                        foldername =  y[0].split('__')[0]
                        datasetname = y[0].split('__')[1]
                        frame = y[0].split('__')[2]
                        print(foldername, datasetname, frame)
                        x_cond = x[:,:3,:,:].cpu()
                        x_gt = x[:,3:,:,:].cpu()

                        print('-x_cond-',x_cond.shape,'-x_gt-',x_gt.shape)
                        # utils.logging.save_image(x_gt, os.path.join(image_folder,foldername, 'gt',sid, f"{frame}.png"))

                        input_res = 224
                        stride = 4
                        # stride = 64

                        
                        utils.logging.save_image(x_cond, os.path.join('results', sid,'input', f"{frame}.png"))


                        h_list = [i for i in range(0, x_cond.shape[2] - input_res + 1, stride)]
                        w_list = [i for i in range(0, x_cond.shape[3] - input_res + 1, stride)]
                        h_list = h_list + [x_cond.shape[2]-input_res]
                        w_list = w_list + [x_cond.shape[3]-input_res]

                        corners = [(i, j) for i in h_list for j in w_list]
                        print('-corners-',len(corners))

                        p_size = input_res
                        x_grid_mask = torch.zeros_like(x_cond).cuda()
                        
                        # print('-x_grid_mask-',x_grid_mask.shape)
                        for (hi, wi) in corners:
                            # print('-hi, wi-',hi, wi)
                            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1
                        x_grid_mask = x_grid_mask.cpu()
                        # et_output = torch.zeros_like(x_cond).cuda()
                        et_output = torch.zeros_like(x_cond).cpu()
                        # print('-et_output -',x_grid_mask.shape,et_output.shape)
                        
                        B,C,H,W = x_cond.shape

                        # manual_batching_size = 4096
                        # manual_batching_size = 3192
                        manual_batching_size = 2048
                        x_cond = x_cond.cuda()

                        torch.cuda.empty_cache()
                        batch_size = 8192
                        batches = [corners[i:i + batch_size] for i in range(0, len(corners), batch_size)]
                        result_patches = []
                        for batch in batches:
                            batch_patches = torch.cat([crop(x_cond, hi, wi, p_size, p_size) for hi, wi in batch], dim=0).cpu()
                            result_patches.append(batch_patches)
                        x_cond_patch = torch.cat(result_patches, dim=0)
                        torch.cuda.empty_cache()

                        outputall = torch.zeros_like(x_cond_patch.cpu())

                        for i in range(0, len(x_cond_patch), manual_batching_size):
                            print(i,i+manual_batching_size, 'using nighthaze model')
                            outputall[i:i+manual_batching_size] = self.nighthaze.Dehaze( data_transform(x_cond_patch[i:i+manual_batching_size]).cuda().float() ).cpu()
                            # outputall[i:i+manual_batching_size] = self.nighthaze.teachermodel( data_transform(x_cond_patch[i:i+manual_batching_size]).cuda().float() ).cpu()
                            # outputall[i:i+manual_batching_size] = self.nighthaze.studentmodel( data_transform(x_cond_patch[i:i+manual_batching_size]).cuda().float() ).cpu()

                            torch.cuda.empty_cache()
                        print('-x_cond_patch -',x_cond_patch.shape,'-outputall -',outputall.shape)
                        et_output = et_output.cuda()
                        for ci in range(len(corners)):
                            hi, wi = corners[ci][0], corners[ci][1]
                            et_output[:, :, hi:hi + p_size, wi:wi + p_size] += outputall[ci*B:(ci+1)*B].cuda()
                        et_output = et_output.cpu()
                            
                            

                        mean_output = torch.div(et_output, x_grid_mask)
                        # print('-mean_output -',mean_output.shape)


                        mean_output = inverse_data_transform(mean_output)
                        torch.cuda.empty_cache()
                        utils.logging.save_image(mean_output, os.path.join('results',sid, 'output', f"{frame}.png"))

                        x_cond = x_cond.cpu()
                        mean_output = mean_output.cpu()
                        all_torchcat = torch.cat((x_cond,mean_output),dim=-1)
                        utils.logging.save_image(all_torchcat, os.path.join('results',sid, 'inp_oup', f"{frame}.png"))