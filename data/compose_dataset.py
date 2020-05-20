# ========================================================
# Compositional GAN
# File for reading the compositional dataset and outputing 
# it in the correct format.
# By Samaneh Azadi
# ========================================================
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
from scipy import misc
from scipy.misc import imread
import torch
import random
import ntpath
import pickle

class ComposeDataset(BaseDataset):
    #taking sample from three distributions A1,A2,B

    def initialize(self, opt):
        self.opt = opt
        with open(opt.datalist) as f:
            # datalist is a txt file containing paths of images in A,B,azi 
            self.info = f.readlines()
        self.A1_paths = sorted([x.strip().split(' ')[0] for x in self.info])
        self.A2_paths = sorted([x.strip().split(' ')[1] for x in self.info])
        self.B_paths = sorted([x.strip().split(' ')[2] for x in self.info])
        rand_s = np.random.permutation(len(self.A1_paths))
        self.A1_paths = [self.A1_paths[i] for i in rand_s]
        self.A2_paths = [self.A2_paths[i] for i in rand_s]
        self.B_paths = [self.B_paths[i] for i in rand_s]

        self.rgb = True if opt.input_nc==3 else False
        self.loadSizeX = opt.loadSizeX
        self.fineSizeX = opt.fineSizeX
        self.loadSizeY = opt.loadSizeY
        self.fineSizeY = opt.fineSizeY

    def __getitem__(self, index):
        B_path = self.B_paths[index]
        A1_path = self.A1_paths[index]
        A2_path = self.A2_paths[index]
        B_root = B_path.split('images')[0]
        B_name = ntpath.basename(B_path)
        M_path = os.path.join(B_root, 'masks', B_name)

        if self.opt.random_view:
            az_diff = int(360/self.opt.num_az)
            az_A1 = int(np.random.permutation(range(0, self.opt.num_az))[0])
            az_A2 = int(np.random.permutation(range(az_A1*az_diff-90, az_A1*az_diff+90, az_diff))[0]/az_diff)
            az_A2 = az_A2%self.opt.num_az
            az_A1 = az_A1%self.opt.num_az
            A1_name =  ntpath.basename(A1_path)
            A2_name =  ntpath.basename(A2_path)
            A1_r_path = '%s/%s-%s.png'%(A1_path,A1_name, az_A1)
            A1_path = '%s/%s-%s.png'%(A1_path,A1_name, az_A2)
            A2_path = '%s/%s-%s.png'%(A2_path,A2_name, az_A2)


        all_B = Image.open(B_path).convert('RGB')
        all_B = all_B.resize((self.opt.loadSizeY*5, self.opt.loadSizeX), Image.BICUBIC)
        all_B = transforms.ToTensor()(all_B)

        w = int(all_B.size(2)/5.0)
        h = all_B.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSizeY - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSizeX - 1))
        offset_center_min_x = int(self.opt.loadSizeX/2.0 - self.opt.fineSizeX/2.0)
        offset_center_max_x = int(self.opt.loadSizeX/2.0 + self.opt.fineSizeX/2.0)
        if self.fineSizeX == self.fineSizeY:
            offset_center_min_y = offset_center_min_x
            offset_center_max_y = offset_center_max_x
        else:
            offset_center_min_y = int(self.opt.loadSizeY/2.0 - self.opt.fineSizeY/2.0)
            offset_center_max_y = int(self.opt.loadSizeY/2.0 + self.opt.fineSizeY/2.0)


        B = all_B[:, h_offset:h_offset + self.opt.fineSizeX,
               w_offset:w_offset + self.opt.fineSizeY]
        B1 = all_B[:, h_offset:h_offset + self.opt.fineSizeX,
               w + w_offset:w + w_offset + self.opt.fineSizeY]
        B2 = all_B[:, offset_center_min_x:offset_center_max_x,
               2*w +offset_center_min_y:2*w +offset_center_max_y]

        all_M = Image.open(M_path)
        all_M = all_M.resize((self.opt.loadSizeY*2, self.opt.loadSizeX), Image.NEAREST)
        all_M = transforms.ToTensor()(all_M)    


        M1 = all_M[0:3, h_offset:h_offset + self.opt.fineSizeX,
               w_offset:w_offset + self.opt.fineSizeY]
        M2 = all_M[0:3, h_offset:h_offset + self.opt.fineSizeX,
               w + w_offset:w + w_offset + self.opt.fineSizeY]

        A1 = Image.open(A1_path).convert('RGB')

        A1 = A1.resize((self.opt.loadSizeY, self.opt.loadSizeX), Image.BICUBIC)
        A1 = transforms.ToTensor()(A1)

        w = A1.size(2)
        h = A1.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSizeY - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSizeX - 1))
        A1 = A1[:, offset_center_min_x:offset_center_max_x,
               offset_center_min_y:offset_center_max_y]

        # rotated input
        if self.opt.random_view:
            A1_r = Image.open(A1_r_path).convert('RGB')

            A1_r = A1_r.resize((self.opt.loadSizeY, self.opt.loadSizeX), Image.BICUBIC)
            A1_r = transforms.ToTensor()(A1_r)

            w = A1_r.size(2)
            h = A1_r.size(1)

            A1_r = A1_r[:, h_offset:h_offset + self.opt.fineSizeX,
                   w_offset:w_offset + self.opt.fineSizeY]


        A2 = Image.open(A2_path).convert('RGB')
        A2 = A2.resize((self.opt.loadSizeY, self.opt.loadSizeX), Image.BICUBIC)
        A2 = transforms.ToTensor()(A2)

        w = A2.size(2)
        h = A2.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSizeY - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSizeX - 1))

        A2 = A2[:, offset_center_min_x:offset_center_max_x,
               offset_center_min_y:offset_center_max_y]
        A1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A1)
        A2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A2)
        B1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B1)
        B2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B2)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
        if self.opt.random_view:
            A1_r = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A1_r)

        if not self.opt.random_view:
            out_dict = {'B': B, 'B1':B1, 'B2':B2, \
                    'M1':M1, 'M2':M2, 'B_paths': B_path, \
                    'A1': A1, 'A2':A2 , 'A_paths':[A1_path, A2_path]}
        else:
            out_dict = {'B': B, 'B1':B1, 'B2':B2,'M1':M1, 'M2':M2, 'B_paths': B_path, \
                    'A1': A1, 'A1_r':A1_r, 'A2':A2 , 'A_paths':[A1_path, A2_path]}

        
        return out_dict

    def __len__(self):
        return len(self.B_paths)

    def name(self):
        return 'ComposeDataset'


class ComposeAlignedDataset(BaseDataset):
    #taking sample from three distributions A1,A2,B

    def initialize(self, opt):
        self.opt = opt
        with open(opt.datalist) as f:
            # datalist is a txt file containing paths of images in A,B,azi 
            self.info = f.readlines()
        self.info = sorted(self.info)
        if opt.phase == 'test' and opt.random_view:
            if not os.path.exists(opt.test_path_azi):
                azi = [random.randrange(-90, 91, 10) for x in range(int(opt.how_many))]
                pickle.dump(azi, open(opt.test_path_azi,'wb'))
            self.test_azi = pickle.load( open( opt.test_path_azi, "rb" ) )
        if opt.random_view:
            self.A_paths = [x.strip().split(' ')[0] for x in self.info]
            self.B_paths = [x.strip().split(' ')[1] for x in self.info]
            self.az_B = [x.strip().split(' ')[2] for x in self.info]
        else:
            self.B_paths = [x.strip().split(' ')[0] for x in self.info]

        self.rgb=True if opt.input_nc==3 else False
        self.loadSizeX = opt.loadSizeX
        self.fineSizeX = opt.fineSizeX
        self.loadSizeY = opt.loadSizeY
        self.fineSizeY = opt.fineSizeY


    def __getitem__(self, index):
        B_path = self.B_paths[index]
        if self.opt.random_view:
            az_B = int(self.az_B[index])
            az_diff = int(360/self.opt.num_az)
            az_B = int(int(az_B/az_diff)*az_diff)
            if hasattr(self, 'test_azi'):
                az_A = int((self.test_azi[index]+az_B)/az_diff)
            else:
                az_A = int(np.random.permutation(range(az_B-90, az_B+90, az_diff))[0]/az_diff)
            az_A = az_A%self.opt.num_az

        AB = Image.open(B_path).convert('RGB')
        AB = AB.resize((self.opt.loadSizeY*5, self.opt.loadSizeX), Image.BICUBIC)
        AB = transforms.ToTensor()(AB)
        
        w_total = AB.size(2)
        w = int(w_total/5)
        h = AB.size(1)

        w_offset = random.randint(0, max(0, w - self.opt.fineSizeY - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSizeX - 1))
        offset_center_min_x = int(self.opt.loadSizeX/2.0 - self.opt.fineSizeX/2.0)
        offset_center_max_x = int(self.opt.loadSizeX/2.0 + self.opt.fineSizeX/2.0)
        if self.fineSizeX == self.fineSizeY:
            offset_center_min_y = offset_center_min_x
            offset_center_max_y = offset_center_max_x
        else:
            offset_center_min_y = int(self.loadSizeY/2.0 - self.fineSizeY/2.0)
            offset_center_max_y = int(self.loadSizeY/2.0 + self.fineSizeY/2.0)

        B = AB[:, h_offset:h_offset + self.fineSizeX,
               w_offset:w_offset + self.fineSizeY]

        if self.opt.phase=='test':
            B1 = AB[:, offset_center_min_x:offset_center_max_x,
                   1*w +offset_center_min_y:1*w +offset_center_max_y]
        else:
            B1 = AB[:, h_offset:h_offset + self.fineSizeX,
                   w + w_offset: w + w_offset + self.fineSizeY]

        B2 = AB[:, offset_center_min_x:offset_center_max_x,
               2*w +offset_center_min_y:2*w +offset_center_max_y]
        

        B1_T = AB[:, h_offset:h_offset + self.fineSizeX,
               3*w + w_offset:3*w + w_offset + self.fineSizeY]
        B2_T = AB[:, h_offset:h_offset + self.fineSizeX,
               4*w + w_offset:4*w + w_offset + self.fineSizeY]

        if self.opt.random_view:

            A_path = self.A_paths[index]
            im_name = ntpath.basename(A_path)
            A_path = '%s/%s-%s.png'%(A_path, im_name, az_A)

            A = Image.open(A_path).convert('RGB')


            A = A.resize((self.loadSizeY, self.loadSizeX), Image.BICUBIC)
            A = transforms.ToTensor()(A)

            w = A.size(2)
            h = A.size(1)
            w_offset = random.randint(0, max(0, w - self.fineSizeY - 1))
            h_offset = random.randint(0, max(0, h - self.fineSizeX - 1))

            A1 = A[:, offset_center_min_x:offset_center_max_x,
                offset_center_min_y:offset_center_max_y]
        else:
            A_path = B_path
            A1 = B1.clone()

        A2 = B2.clone()
        
        A1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A1)
        A2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A2)
        B1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B1)
        B2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B2)
        B1_T = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B1_T)
        B2_T = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B2_T)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if not self.rgb:
            B = torch.mean(B, 0).unsqueeze(0)
            A1 = torch.mean(A1, 0).unsqueeze(0)
            A2 = torch.mean(A2, 0).unsqueeze(0)
            B1 = torch.mean(B1, 0).unsqueeze(0)
            B2 = torch.mean(B2, 0).unsqueeze(0)
            B1_T = torch.mean(B1_T, 0).unsqueeze(0)
            B2_T = torch.mean(B2_T, 0).unsqueeze(0)


        out_dict = {'B': B, 'B_paths': B_path,'B1':B1,'B2':B2,'B1_T':B1_T,'B2_T':B2_T, 'A1': A1, 'A2':A2, \
                'A_paths':[A_path, B_path]}

        return out_dict

    def __len__(self):
        return len(self.B_paths)

    def name(self):
        return 'ComposeAlignedDataset'
