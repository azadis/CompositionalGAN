# ========================================================
# Compositional GAN
# File for reading the AFN dataset and outputing 
# it in the correct format.
# By Samaneh Azadi
# ========================================================

import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from PIL import Image
import numpy as np
import torch
import random
import ntpath

class AFNDataset(BaseDataset):
    ## (input, target, relative transformation)
    def initialize(self, opt):
        self.opt = opt

        with open(opt.datalist) as f:
            # datalist is a txt file containing all paths 
            self.A_paths = f.readlines()
        self.A_paths = sorted([x.strip() for x in self.A_paths]) 

        with open(opt.masklist) as f:
            # datalist is a txt file containing all paths 
            self.A_mask_paths = f.readlines()
        self.A_mask_paths = sorted([x.strip() for x in self.A_mask_paths] )

        self.transformer = transforms.Compose([
            transforms.Resize((opt.loadSize, opt.loadSize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        self.transformer_m = transforms.Compose([
            transforms.Resize((opt.loadSize, opt.loadSize)),
            transforms.ToTensor()])

        self.fineSize = opt.fineSize

        transitions = np.arange(-180, 0, self.opt.view_step).tolist() + np.arange(0, 181, self.opt.view_step).tolist()
        self.num_trans = len(transitions)
        self.onehot_dict = {}
        for i in range(self.num_trans):
            self.onehot_dict[transitions[i]] = i



    def __getitem__(self, index):
        
        A_path = self.A_paths[index]
        A_mask_path = self.A_mask_paths[index]

        name = ntpath.basename(A_path)
        parent_dir = A_path.split(name)[0]
        # img_id = name.split('-')[0]
        az = range(self.opt.num_az)
        az_inp = int(name.split('-')[-1].split('.')[0])
        img_id = '-'.join(name.split('-')[:-1])

        # az_out = int(np.random.permutation(list(set(az)-set([int(az_inp)])))[0])
        az_out = int(np.random.permutation(list(set(az)))[0])
        B_path = os.path.join(parent_dir,'%s-%s.png'%(img_id,str(az_out)))


        parent_dir = A_mask_path.split(img_id)[0]
        mask_ext = A_mask_path.split('.')[-1]
        B_mask_path = os.path.join(parent_dir,img_id, '%s-%s.%s'%(img_id,str(az_out), mask_ext))

        az_inp *= 10
        az_out *= 10
        if az_inp > 180:
            az_inp -= 360


        if az_out > 180:
            az_out -= 360

        relative_degree = az_out - az_inp
        relative_degree //= self.opt.view_step
        relative_degree = int(relative_degree)*self.opt.view_step

        if relative_degree > 180:
            relative_degree -= 360
        elif relative_degree < -180:
            relative_degree += 360
        relative_trans = np.zeros((self.num_trans,1))

        relative_trans[self.onehot_dict[relative_degree]] = 1

        A = Image.open(A_path).convert('RGB')
        A = self.transformer(A)
        
        B = Image.open(B_path).convert('RGB')
        B = self.transformer(B)

        
        w = A.size(2)
        h = A.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]

        B = B[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]

        A_mask = Image.open(A_mask_path)
        A_mask = self.transformer_m(A_mask).repeat(3,1,1)

        B_mask = Image.open(B_mask_path)
        B_mask = self.transformer_m(B_mask).repeat(3,1,1)

        # B_mask = (torch.mean(B, dim=0, keepdim=True) < 1).repeat(3,1,1).type(torch.FloatTensor)
        # A_mask = (torch.mean(A, dim=0, keepdim=True) < 1).repeat(3,1,1).type(torch.FloatTensor)

        out_dict = {'B': B, 'B_paths': B_path, 'mask_B': B_mask,\
                    'A': A, 'A_paths': A_path, 'mask_A': A_mask,\
                    'trans':relative_trans}

        return out_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AFNDataset'

