# ========================================================
# Compositional GAN
# File for reading the Relative AFN dataset and outputing 
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

class AFNComposeDataset(BaseDataset):
    ## (input, target, relative transformation)
    def initialize(self, opt):
        self.opt = opt
        with open(opt.datalist) as f:
            # datalist is a txt file containing all paths 
            self.A_paths = f.readlines()
        self.A1_paths = ([x.strip().split(' ')[0] for x in self.A_paths]) 
        self.A2_paths = ([x.strip().split(' ')[1] for x in self.A_paths]) 

        self.transformer = transforms.Compose([
            transforms.Resize((opt.loadSize, opt.loadSize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.fineSize = opt.fineSize

        self.max_diff = int(self.opt.num_az/4)
        self.min_diff = int(-self.opt.num_az/4)

        self.az = list(range(0, self.opt.num_az))

    def __getitem__(self, index):

        A1_path = self.A1_paths[index]
        A2_path = self.A2_paths[index]

        name_A1 = ntpath.basename(A1_path)
        parent_dir_A1 = A1_path.split(name_A1)[0]

        name_A2 = ntpath.basename(A2_path)
        parent_dir_A2 = A2_path.split(name_A2)[0]

        az = np.random.permutation(self.az)
        az_inp = az[0]

        #rotate object1 no more than 180o (to the closet correct pose)
        az_o = np.random.permutation([az_inp+i for i in range(self.min_diff, self.max_diff)])
        az_out = az_o[0] 
        az_out = az_out%self.opt.num_az


        A1_path = os.path.join(parent_dir_A1,'%s'%name_A1, '%s-%s.png'%(name_A1, str(az_inp)))
        #A2 and B in the same view point (with 180o difference)
        B_path = os.path.join(parent_dir_A1,'%s'%name_A1, '%s-%s.png'%(name_A1, str(az_out)))
        az_out_t = int(az_out-self.opt.num_az/2)%self.opt.num_az
        A2_path = os.path.join(parent_dir_A2,'%s'%name_A2, '%s-%s.png'%(name_A2, str(az_out_t)))


        A1 = Image.open(A1_path).convert('RGB')
        A1 = self.transformer(A1)
        A2 = Image.open(A2_path).convert('RGB')
        A2 = self.transformer(A2)
        A2 = ((torch.mean(A2, dim=0, keepdim=True) <1)*1).type(torch.FloatTensor)
        
        B = Image.open(B_path).convert('RGB')
        B = self.transformer(B)

        
        w = A1.size(2)
        h = A1.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A1 = A1[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]

        A2 = A2[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]

        B = B[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        out_dict = {'B': B, 'B_paths': B_path,\
                    'A1': A1, 'A2':A2, 'A_paths': [A1_path,A2_path]\
                    }

        return out_dict

    def __len__(self):
        return len(self.A1_paths)

    def name(self):
        return 'AFNComposeDataset'

