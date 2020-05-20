# ========================================================
# Compositional GAN
# Model for the Unpaired Training Data
# By Samaneh Azadi
# ========================================================


import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch import index_select, LongTensor, nn
from scipy import misc
from scipy.ndimage.morphology import binary_erosion
import torch.nn.functional as F


class objComposeUnsuperviseModel(BaseModel):
    def name(self):
        return 'objComposeUnsuperviseModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.y_x = int(float(opt.fineSizeY)/opt.fineSizeX)
        # -------------------------------
        # Define Networks
        # -------------------------------
        # Composition Generator
        self.netG_comp = networks.define_G(2*opt.output_nc, opt.input_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      opt.noise, y_x=self.y_x)
        if self.opt.img_completion:
            #inpainting network
            self.netG1_completion = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                          opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                          opt.noise, y_x=self.y_x)
            self.netG2_completion = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                          opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                          opt.noise, y_x=self.y_x)
        if opt.lambda_mask:
            opt.which_model_netG = '%s_masked'%opt.which_model_netG

        # Decomposition Generator
        self.netG_decomp = networks.define_G(opt.input_nc, 2*opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      opt.noise, y_x=self.y_x)



        if opt.random_view:
            # Relative Appearance Flow Network
            self.netAFN = networks.define_AFN(opt.which_model_AFN, input_nc=opt.input_nc+1,init_type=opt.init_type, gpu_ids=self.gpu_ids)
            if self.isTrain and not opt.continue_train:
                self.load_network(self.netAFN, 'AFN', opt.which_epoch_AFN)
        #Spatial Transformer networks
        self.netSTN_dec =  networks.define_STN(2*opt.output_nc, opt.fineSizeX, self.gpu_ids, y_x=self.y_x, STN_model=opt.STN_model)
        self.netSTN_c =  networks.define_STN(2*opt.output_nc, opt.fineSizeX, self.gpu_ids, y_x=self.y_x, STN_model=opt.STN_model)



        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if self.opt.conditional:
                inp_disc = 2*opt.output_nc
            else:
                inp_disc = opt.output_nc
            #Discriminator Networks
            self.netD_A1 = networks.define_D(inp_disc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,y_x=self.y_x)
        
            self.netD_A2 = networks.define_D(inp_disc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,y_x=self.y_x)
            if self.opt.img_completion:
                self.netD1_completion = networks.define_D(inp_disc, opt.ndf,
                                              opt.which_model_netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,y_x=self.y_x)
                self.netD2_completion = networks.define_D(inp_disc, opt.ndf,
                                              opt.which_model_netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,y_x=self.y_x)            

            if opt.conditional:
                in_ch = opt.input_nc*3
            else:
                in_ch = opt.input_nc
            self.netD_B = networks.define_D(in_ch, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,y_x=self.y_x)



        # ---------------------------------
        # Load networks
        #----------------------------------
        if not self.isTrain or opt.continue_train:
            # if opt.phase != 'test': 
            if opt.random_view:
                self.model_names += ['AFN']
                self.epoch_labels += [opt.which_epoch_AFN]
            if self.opt.img_completion and (int(opt.which_epoch_completion)+int(opt.which_epoch))>0:
                self.model_names +=  ['G1_completion','G2_completion']
                self.epoch_labels += [str(int(opt.which_epoch_completion)+int(opt.which_epoch))]*2

            if int(opt.which_epoch):
                self.model_names += ['G_decomp','G_comp']
                self.epoch_labels += [opt.which_epoch]*2
            if int(opt.which_epoch)+int(opt.which_epoch_STN):
                self.model_names += ['STN_dec','STN_c']
                self.epoch_labels += [str(int(opt.which_epoch)+int(opt.which_epoch_STN))]*2


            if (self.isTrain) and (opt.phase != 'test'):
                if int(opt.which_epoch):
                    self.model_names += ['D_A1','D_A2','D_B']
                    self.epoch_labels += [opt.which_epoch]*3


                if self.opt.img_completion and int(self.opt.which_epoch_completion):                    
                    self.model_names += ['D1_completion','D2_completion']
                    self.epoch_labels += [str(int(opt.which_epoch_completion)+int(opt.which_epoch))]*2

        else:
            if self.opt.img_completion and int(self.opt.which_epoch_completion) >0:
                self.model_names += ['G1_completion','G2_completion','D1_completion','D2_completion']
                self.epoch_labels += [opt.which_epoch_completion]*4
        self.load_networks()

        # ---------------------------------
        # initialize optimizers
        # ---------------------------------
        self.model_names = ['G_decomp','G_comp','STN_dec','STN_c']
        if self.opt.img_completion:
            self.model_names += ['G1_completion','G2_completion']
        if self.isTrain:
            self.model_names += ['D_A1','D_A2','D_B']
            if self.opt.img_completion:
                self.model_names += ['D1_completion','D2_completion']
        if opt.random_view:
            self.model_names += ['AFN']
        
    
        if self.isTrain:
            self.schedulers = []
            self.set_optimizers()
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))
            self.print_networks(verbose=False)

            self.fake_A1_pool = ImagePool(opt.pool_size)
            self.fake_A2_pool = ImagePool(opt.pool_size)
            self.fake_BA1_pool = ImagePool(opt.pool_size)
            self.fake_BA2_pool = ImagePool(opt.pool_size)

            # ----------------------------------
            # define loss functions
            # ----------------------------------
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionCLS = nn.CrossEntropyLoss()
            self.criterionbCLS = nn.BCELoss()

            # loss function initializations
            self.loss_G_GAN = Variable(torch.Tensor(1).fill_(0).cuda())
            self.loss_D_real = Variable(torch.Tensor(1).fill_(0).cuda())
            self.loss_D_fake = Variable(torch.Tensor(1).fill_(0).cuda())
            self.loss_G_L1 = Variable(torch.Tensor(1).fill_(0).cuda())
            self.loss_G_seg = Variable(torch.Tensor(1).fill_(0).cuda())
            self.loss_gp = Variable(torch.Tensor(1).fill_(0).cuda())
            self.loss_AFN = Variable(torch.Tensor(1).fill_(0).cuda())
            self.loss_G_mask = Variable(torch.Tensor(1).fill_(0).cuda())
            self.loss_STN = Variable(torch.Tensor(1).fill_(0).cuda())
            self.loss_G_completion = Variable(torch.Tensor(1).fill_(0).cuda())
            self.loss_D_completion = Variable(torch.Tensor(1).fill_(0).cuda())
        self.softmax = torch.nn.Softmax(dim=1)



    def set_input_train(self, input):
        '''samples of real distribution (from training set) to be used at test time'''

        self.ex_B = input['B']
        self.ex_B1 = input['B1']
        self.ex_B2 = input['B2']
        self.input_M1 = input['M1']
        self.input_M2 = input['M2']
        input_vars = ['ex_B', 'ex_B1', 'ex_B2',
                        'input_M1', 'input_M2']
        if len(self.gpu_ids) > 0:
            self.tocuda(input_vars)
        for name in input_vars:
            if isinstance(name, str):
                var = getattr(self, name)
                setattr(self, name, Variable(var))

        self.ex_B1_T = torch.mul(self.ex_B, self.input_M1) + (1-self.input_M1)
        self.ex_B2_T = torch.mul(self.ex_B, self.input_M2) + (1-self.input_M2)
        self.ex_B1_T = Variable(self.ex_B1_T.data, requires_grad=False)
        self.ex_B2_T = Variable(self.ex_B2_T.data, requires_grad=False)


    def set_input_test(self, input):
        ''' samples at test time; no target composite image given'''

        self.input_A1 = input['A1']
        self.input_A2 = input['A2']
        self.A_paths = input['A_paths']
        input_vars = ['input_A1', 'input_A2']

        if len(self.gpu_ids) > 0:
            self.tocuda(input_vars)

        if self.opt.random_view:
            self.input_A1_r = self.input_A1


    def set_input(self, input):
        '''Samples at training time'''

        self.input_A1 = input['A1']
        self.input_A2 = input['A2']
        self.input_B1 = input['B1']
        self.input_B2 = input['B2']
        self.input_M1 = input['M1']
        self.input_M2 = input['M2']
        self.A_paths = input['A_paths']
        if self.opt.random_view:
            self.input_A1_r = input['A1_r']

        self.input_B = input['B']
        self.B_paths = input['B_paths']

        input_vars = ['input_A1', 'input_A2', 'input_B',
                    'input_B1', 'input_B2', 'input_M1', 'input_M2']
        if self.opt.random_view:
            input_vars += ['input_A1_r']

        if len(self.gpu_ids) > 0:
            self.tocuda(input_vars)

        self.real_M1_s = Variable(self.input_M1[:,0:3,:,:])
        self.real_M2_s = Variable(self.input_M2[:,0:3,:,:])

        self.real_M = (self.real_M1_s + self.real_M2_s*2).type(torch.cuda.LongTensor)
        self.real_M = self.real_M[:,0,:,:]

    def forward_STN(self):
        '''Forward pass for the spatial transformer network'''

        self.real_B = Variable(self.input_B)
        self.real_B1 = Variable(self.input_B1)
        self.real_B2 = Variable(self.input_B2)
        self.real_B1_T = torch.mul(self.real_B, self.real_M1_s) + (1-self.real_M1_s)
        self.real_B2_T = torch.mul(self.real_B, self.real_M2_s) + (1-self.real_M2_s)
        self.real_B1_T = Variable(self.real_B1_T.data, requires_grad=False)
        self.real_B2_T = Variable(self.real_B2_T.data, requires_grad=False)

        self.stn_B1_T , self.stn_B2_T = self.netSTN_c(torch.cat((self.real_B1, self.real_B2),1))
        self.stn_B1 , self.stn_B2 = self.netSTN_dec(torch.cat((self.real_B1_T, self.real_B2_T),1))



    def forward_inpainting(self):
        '''forward pass for the inpainting network '''

        self.real_A1 = Variable(self.input_A1)
        self.real_A2 = Variable(self.input_A2)

        #affine transformation of each of the inputs wrt the other one
        self.real_A1_T, self.real_A2_T = self.netSTN_c(torch.cat((self.real_A1, self.real_A2),1))
        self.real_A1_T = Variable(self.real_A1_T.data, requires_grad=False)
        self.real_A2_T = Variable(self.real_A2_T.data, requires_grad=False)

        self.mask_A1_T = torch.mean(self.real_A1_T,dim=1,keepdim=True)
        self.mask_A2_T = torch.mean(self.real_A2_T,dim=1,keepdim=True)
        self.mask_A1_T = (self.mask_A1_T<0.9*torch.max(self.mask_A1_T).data[0]).type(torch.cuda.FloatTensor)
        self.mask_A2_T = (self.mask_A2_T<0.9*torch.max(self.mask_A2_T).data[0]).type(torch.cuda.FloatTensor)
        self.mask_A1_T = self.mask_A1_T.repeat(1,3,1,1)
        self.mask_A2_T = self.mask_A2_T.repeat(1,3,1,1)
        
        #zero out part of OBJ1 given the mask of OBJ2 (and similarly for OBJ1 )
        self.segment_A1 = torch.mul(1-self.mask_A2_T, self.real_A1_T) + (self.mask_A2_T)
        self.segment_A2 = torch.mul(1-self.mask_A1_T, self.real_A2_T) + (self.mask_A1_T)
        self.segment_A1 = Variable(self.segment_A1.data, requires_grad=False)
        self.segment_A2 = Variable(self.segment_A2.data, requires_grad=False)

        #Pass each masked object to its inpainting network
        self.fake_A1_compl = self.netG1_completion(self.segment_A1)
        self.fake_A2_compl = self.netG2_completion(self.segment_A2)



    def forward_test(self):
        '''starting from full OBJ1 and OBJ2 @ test time'''

        self.real_A1 = Variable(self.input_A1)
        self.real_A2 = Variable(self.input_A2)
        if self.opt.random_view:
            self.real_A1_r = Variable(self.input_A1_r)
            self.mask_A2 = (torch.mean(self.real_A2,dim=1,keepdim=True)<1).type(torch.cuda.FloatTensor)


        #start the cycle from real_A1, real_A2
        #AFN
        if self.opt.random_view:
            self.flow, self.mask_pred = self.netAFN(torch.cat((self.real_A1_r, self.mask_A2),1))
            self.flow = self.flow.permute(0,2,3,1)
            # grid values should be in the range [-1,1]
            self.fake_A1 = F.grid_sample(self.real_A1_r, self.flow)
        else:
            self.fake_A1 = self.real_A1
        self.fake_A2 = self.real_A2

        #STN
        self.fake_A1_T, self.fake_A2_T = (self.netSTN_c(torch.cat((self.fake_A1.detach(),self.fake_A2),1)))
        self.mask_A1_T = torch.mean(self.fake_A1_T,dim=1,keepdim=True)
        self.mask_A2_T = torch.mean(self.fake_A2_T,dim=1,keepdim=True)
        self.mask_A1_T = (self.mask_A1_T<self.opt.Thresh1*torch.max(self.mask_A1_T).data[0]).type(torch.cuda.FloatTensor)
        self.mask_A2_T = (self.mask_A2_T<self.opt.Thresh2*torch.max(self.mask_A2_T).data[0]).type(torch.cuda.FloatTensor)
        # print(self.mask_A1_T.size(),self.mask_A1_T[0,0,:,:].data.cpu().numpy().shape)


        if self.opt.erosion:
            #erode borders of gt masks to have clean outputs
            self.mask_A1_T = self.mask_A1_T.data.cpu().numpy().astype(int)
            self.mask_A2_T = self.mask_A2_T.data.cpu().numpy().astype(int)
            self.mask_A1_T = Variable(torch.Tensor(1*binary_erosion(self.mask_A1_T, structure=np.ones((3,3))[np.newaxis,np.newaxis,:,:])).cuda())
            self.mask_A2_T = Variable(torch.Tensor(1*binary_erosion(self.mask_A2_T, structure=np.ones((3,3))[np.newaxis,np.newaxis,:,:])).cuda())
            self.mask_A1_T = self.mask_A1_T.type(torch.cuda.FloatTensor)
            self.mask_A2_T = self.mask_A2_T.type(torch.cuda.FloatTensor)


        
        if self.opt.G1_completion:
            mask_A2_T = self.mask_A2_T.repeat(1,3,1,1)
            masked_fake_A1 = torch.mul(1-mask_A2_T, self.fake_A1_T) + (mask_A2_T)
            fake_A1_T = (torch.mul(1-mask_A2_T,self.fake_A1_T) +
                        torch.mul(mask_A2_T, self.netG1_completion(masked_fake_A1)))
        else:
            fake_A1_T = self.fake_A1_T
        if self.opt.G2_completion:
            mask_A1_T = self.mask_A1_T.repeat(1,3,1,1)
            masked_fake_A2 = torch.mul(1-mask_A1_T, self.fake_A2_T) + mask_A1_T
            fake_A2_T = (torch.mul(1-mask_A1_T,self.fake_A2_T) +
                torch.mul(mask_A1_T, self.netG2_completion(masked_fake_A2)))

        else:
            fake_A2_T = self.fake_A2_T

        self.fake_A = torch.cat((fake_A1_T.detach(),fake_A2_T.detach()),1)

        #Composition network
        self.fake_B = self.netG_comp(self.fake_A)

        # Decomposition and mask prediction networks
        if self.opt.lambda_mask:
            self.B1_B2, self.M1_M2 = self.netG_decomp(self.fake_B)
            self.M1_M2_normal = self.softmax(self.M1_M2)
            v,m = torch.max(self.M1_M2_normal, dim=1, keepdim=True)
            self.fake_M1_s = ((m==1)*1).type(torch.cuda.FloatTensor)
            self.fake_M2_s = ((m==2)*1).type(torch.cuda.FloatTensor)

        else:
            self.B1_B2 = self.netG_decomp(self.fake_B)

        self.fake_B.retain_grad()
        self.fake_B1_T = index_select(self.B1_B2,1,Variable(LongTensor(range(0,self.opt.input_nc)).cuda()))
        self.fake_B2_T = index_select(self.B1_B2,1,Variable(LongTensor(range(self.opt.input_nc,2*self.opt.input_nc)).cuda()))

        #STN: return objects to the center
        self.fake_B1,self.fake_B2 = self.netSTN_dec(torch.cat((self.fake_B1_T, self.fake_B2_T),1))

        self.real_M0_s = 1 - (((self.mask_A1_T + self.mask_A2_T)>=1)*1).type(torch.cuda.FloatTensor)

        #apply predicted mask on gt full mask
        self.M1_s_only = torch.mul(1-self.mask_A2_T, self.mask_A1_T)
        self.M2_s_only = torch.mul(1-self.mask_A1_T, self.mask_A2_T)
        self.real_M1_s = self.mask_A1_T
        self.real_M2_s = self.mask_A2_T

        if self.opt.lambda_mask:
            self.real_M1_s = torch.mul(self.real_M1_s, self.fake_M1_s)
            self.real_M2_s = torch.mul(self.real_M2_s, self.fake_M2_s)

        missed_overlp = (((self.real_M1_s + self.real_M2_s)==0)*1).type(torch.cuda.FloatTensor)
        self.missed_A1 = torch.mul(missed_overlp, self.M1_s_only)
        self.missed_A2 = torch.mul(missed_overlp, self.M2_s_only)
        self.real_M1_s = self.missed_A1 + self.real_M1_s
        self.real_M2_s = self.missed_A2 + self.real_M2_s


        self.real_M1_s = self.real_M1_s.repeat(1,3,1,1)
        self.real_M2_s = self.real_M2_s.repeat(1,3,1,1)
        self.real_M0_s = self.real_M0_s.repeat(1,3,1,1)

        self.real_M = (self.real_M1_s + self.real_M2_s*2).type(torch.cuda.LongTensor)
        self.real_M = self.real_M[:,0,:,:]


        if self.opt.phase=='test':
            if self.opt.random_view:
                self.mask_pred = (self.mask_pred > 0.5).float() * 1

        # fake_B obtained by direct summation
        self.fake_B_sum = (torch.mul(self.real_M1_s, self.fake_A1_T) + torch.mul(self.real_M2_s, self.fake_A2_T)
                            + (1-self.real_M1_s - self.real_M2_s))

    def forward(self):
        '''starting from segments of input_B @ training time'''

        self.real_A1 = Variable(self.input_B1)
        self.real_A2 = Variable(self.input_B2)
        self.real_B = Variable(self.input_B) 


        self.real_A1_T = torch.mul(self.real_B, self.real_M1_s) + (1-self.real_M1_s)
        self.real_A2_T = torch.mul(self.real_B, self.real_M2_s) + (1-self.real_M2_s)
        
        if self.opt.G1_completion:
            self.real_A1_T = self.netG1_completion(self.real_A1_T)
        if self.opt.G2_completion:
            self.real_A2_T = self.netG2_completion(self.real_A2_T)
        self.real_A1_T = Variable(self.real_A1_T.data, requires_grad=False)
        self.real_A2_T = Variable(self.real_A2_T.data, requires_grad=False)
        self.real_B1_T = self.real_A1_T
        self.real_B2_T = self.real_A2_T

        if self.opt.G1_completion:
            self.fake_A1 = self.netG1_completion(self.real_A1)
        else:
            self.fake_A1 = self.real_A1
        if self.opt.G2_completion:
            self.fake_A2 = self.netG2_completion(self.real_A2)
        else:
            self.fake_A2 = self.real_A2
        self.real_B1 = Variable(self.fake_A1.data, requires_grad=False)
        self.real_B2 = Variable(self.fake_A2.data, requires_grad=False)


        #Composition network
        self.fake_A1 = Variable(self.fake_A1.data, requires_grad=False)
        self.fake_A2 = Variable(self.fake_A2.data, requires_grad=False)
        self.fake_A1_T, self.fake_A2_T = (self.netSTN_c(torch.cat((self.fake_A1,self.fake_A2),1)))
        self.fake_A = torch.cat((self.fake_A1_T,self.fake_A2_T),1)
        self.fake_B = self.netG_comp(self.fake_A)


        # Decomposition and mask prediction networks
        if self.opt.lambda_mask:
            self.B1_B2, self.M1_M2 = self.netG_decomp(self.fake_B)
            self.M1_M2_normal = self.softmax(self.M1_M2)
            v,m = torch.max(self.M1_M2_normal, dim=1, keepdim=True)
            self.fake_M1_s = ((m==1)*1).type(torch.cuda.FloatTensor)
            self.fake_M2_s = ((m==2)*1).type(torch.cuda.FloatTensor)

        else:
            self.B1_B2 = self.netG_decomp(self.fake_B)


        self.fake_B.retain_grad()

        self.fake_B1_T = index_select(self.B1_B2,1,Variable(LongTensor(range(0,self.opt.input_nc)).cuda()))
        self.fake_B2_T = index_select(self.B1_B2,1,Variable(LongTensor(range(self.opt.input_nc,2*self.opt.input_nc)).cuda()))

        #STN: return objects back to center
        self.fake_B1, self.fake_B2 = self.netSTN_dec(torch.cat((self.fake_B1_T, self.fake_B2_T),1))

        #fakeB as a direct sum 
        self.fake_B_sum = (torch.mul(self.real_M1_s, self.fake_A1_T) + torch.mul(self.real_M2_s, self.fake_A2_T)
                            + (1-self.real_M1_s - self.real_M2_s))

    # get image paths
    def get_image_paths(self):
        if self.opt.phase == 'test':
            return self.A_paths[0]
        else:
            return self.B_paths

    def calc_gradient_penalty(self,netD, real, fake):
        '''Gradient Penalty'''
        alpha = self.Tensor(real.size(0), 1, 1, 1).uniform_()
        alpha = alpha.expand(real.size())
        mixed = Variable(alpha * real.data + (1 - alpha) * fake.data, requires_grad=True)
        pred = netD.forward(mixed)
        grad = torch.autograd.grad(outputs=pred, inputs=mixed, grad_outputs=torch.ones(pred.size()).cuda(0),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad = grad.view(real.size(0), -1)
        loss_gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return loss_gp


    def backward_STN(self):
        '''backward pass for training STN networks only'''
        self.loss_STN = self.criterionL1(self.stn_B1, self.real_B1) + self.criterionL1(self.stn_B2, self.real_B2)
        self.loss_STN += 100*(self.criterionL1(self.stn_B1_T, self.real_B1_T) + self.criterionL1(self.stn_B2_T, self.real_B2_T))
        self.loss_STN.backward()


    def backward_G_completion(self):
        '''Backward pass for the generator in training the inpainting networks only'''

        loss_G_GAN = 0 
        self.loss_G_completion = 0

        if self.opt.G1_completion:
            # if inpainting OBJ1:
            fake_A1 = torch.cat((self.segment_A1, self.fake_A1_compl),1)
            pred_fake_A1 = self.netD1_completion(fake_A1)
            #GAN loss
            loss_G_GAN = self.criterionGAN(pred_fake_A1, True)
            #L1 pixel loss
            self.loss_G_completion = self.opt.lambda_L2 * self.criterionL1(self.fake_A1_compl, self.real_A1_T)
            


        if self.opt.G2_completion:
            #if inpainting OBJ2:
            fake_A2 = torch.cat((self.segment_A2, self.fake_A2_compl),1)
            pred_fake_A2 = self.netD2_completion(fake_A2)
            #GAN loss
            loss_G_GAN += self.criterionGAN(pred_fake_A2, True)
            #L1 pixel loss
            self.loss_G_completion += self.opt.lambda_L2 * self.criterionL1(self.fake_A2_compl, self.real_A2_T)

        self.loss_G_completion += self.opt.lambda_gan * loss_G_GAN        
        self.loss_G_completion.backward()


    def backward_D_completion(self): 
        '''Backward pass for the discriminator in training the inpainting networks only'''
           
        fake_A1 = torch.cat((self.segment_A1, self.fake_A1_compl),1)
        real_A1 = torch.cat((self.segment_A1, self.real_A1_T),1)

        fake_A1 = self.fake_A1_pool.query(fake_A1.data)
        pred_fake_A1 = self.netD1_completion(fake_A1.detach())
        #real
        pred_real_A1 = self.netD1_completion(real_A1)


        fake_A2 = torch.cat((self.segment_A2, self.fake_A2_compl),1)
        real_A2 = torch.cat((self.segment_A2, self.real_A2_T),1)

        fake_A2 = self.fake_A2_pool.query(fake_A2.data)
        pred_fake_A2 = self.netD2_completion(fake_A2.detach())
        #real
        pred_real_A2 = self.netD2_completion(real_A2)

        loss_D_fake = self.criterionGAN(pred_fake_A1, False) + self.criterionGAN(pred_fake_A2, False)
        loss_D_real = self.criterionGAN(pred_real_A1, True) + self.criterionGAN(pred_real_A2, True)
        loss_gp_A1 = self.calc_gradient_penalty(self.netD1_completion, real_A1.detach(), fake_A1.detach())
        loss_gp_A2 = self.calc_gradient_penalty(self.netD2_completion, real_A2.detach(), fake_A2.detach())
        self.loss_D_completion = 0.5*(loss_D_fake + loss_D_real) + self.opt.lambda_gp * (loss_gp_A1 + loss_gp_A2)

        self.loss_D_completion.backward()

    def backward_G(self):
        '''backward pass for the generator in training the unsupervised model'''
        self.loss_G = 0 

        #GAN Loss
        if self.opt.lambda_gan:
            if self.opt.conditional:
                inp_D_fake = self.fake_B

                pred_fake_B1 = self.netD_A1(torch.cat((inp_D_fake.detach(), self.fake_B1_T),1))
                pred_fake_B2 = self.netD_A2(torch.cat((inp_D_fake.detach(), self.fake_B2_T),1))
            else:
                pred_fake_B1 = self.netD_A1(self.fake_B1_T)
                pred_fake_B2 = self.netD_A2(self.fake_B2_T)

            self.loss_G_GAN_B1 = self.criterionGAN(pred_fake_B1, True)
            self.loss_G_GAN_B2 = self.criterionGAN(pred_fake_B2, True)
            

            if self.opt.conditional:
                inp_pair = torch.cat((self.fake_A1_T, self.fake_A2_T),1)

                pred_fake_B = self.netD_B(torch.cat((inp_pair.detach(), self.fake_B),1))
            else:
                pred_fake_B = self.netD_B(self.fake_B)
            self.loss_G_GAN_B = self.criterionGAN(pred_fake_B,True)

            self.loss_G_GAN  = self.opt.lambda_gan * (self.loss_G_GAN_B)
            if self.opt.decomp:
                self.loss_G_GAN += self.opt.lambda_gan * (self.loss_G_GAN_B1 + self.loss_G_GAN_B2)
                self.loss_G_GAN /= 3.0

            self.loss_G +=  self.loss_G_GAN
        
        # pixel L1 loss
        self.loss_G_L1 = 0.5*(self.criterionL1(self.fake_B1_T, self.real_A1_T.detach()) +
                          self.criterionL1(self.fake_B2_T, self.real_A2_T.detach())) 

        self.loss_G_L1 += self.criterionL1(self.fake_B, self.real_B)
        self.loss_G_L1 += 0.5*(self.criterionL1(self.fake_A1_T, self.real_A1_T.detach()) +
                          self.criterionL1(self.fake_A2_T, self.real_A2_T.detach())) 
        self.loss_G += self.opt.lambda_L2 * self.loss_G_L1 

        # Mask loss
        self.loss_G_mask = self.criterionCLS(self.M1_M2, self.real_M)
        self.loss_G +=  self.opt.lambda_mask * self.loss_G_mask

        self.loss_G.backward()


    def backward_D(self, AorB='B'):
        '''backward pass for the discriminator in training the unpaired model'''

        # Fake
        # stop backprop to the generator by detaching fake_B
        if self.opt.conditional:
            if AorB == 'A':
                inp_D_fake = self.fake_B
                inp_D_real = self.real_B
            else:
                inp_D_fake = self.real_B
                inp_D_real = self.real_B
            
            fake_B1 = torch.cat((inp_D_fake, self.fake_B1_T),1)
            fake_B2 = torch.cat((inp_D_fake, self.fake_B2_T),1)
            real_B1 = torch.cat((inp_D_real, self.real_B1_T),1)
            real_B2 = torch.cat((inp_D_real, self.real_B2_T),1)
        else:
            fake_B1 = self.fake_B1_T
            fake_B2 = self.fake_B2_T
            real_B1 = self.real_B1_T
            real_B2 = self.real_B2_T

        fake_BA1 = self.fake_BA1_pool.query(fake_B1.data)
        fake_BA2 = self.fake_BA2_pool.query(fake_B2.data)
        pred_fake_B1 = self.netD_A1(fake_BA1.detach())
        pred_fake_B2 = self.netD_A2(fake_BA2.detach())
        #real
        pred_real_B1 = self.netD_A1(real_B1)
        pred_real_B2 = self.netD_A2(real_B2)

        self.loss_D_fake_B1 = self.criterionGAN(pred_fake_B1, False)
        self.loss_D_real_B1 = self.criterionGAN(pred_real_B1,True)
        self.loss_D_fake_B2 = self.criterionGAN(pred_fake_B2,False)        
        self.loss_D_real_B2 = self.criterionGAN(pred_real_B2,True)


        if self.opt.conditional:
            if AorB == 'A':
                fake_B = torch.cat((self.fake_A1_T, self.fake_A2_T, self.fake_B),1)
            else:
                fake_B = torch.cat((self.fake_B1_T, self.fake_B2_T, self.fake_B),1)
            real_B = torch.cat((self.real_B1_T, self.real_B2_T, self.real_B),1)

        else:
            fake_B = self.fake_B
            real_B = self.real_B

        pred_fake_B = self.netD_B(fake_B.detach())
        pred_real_B = self.netD_B(real_B.detach())

        self.loss_D_fake_B = self.criterionGAN(pred_fake_B,False)

        self.loss_D_real_B = self.criterionGAN(pred_real_B, True)

        self.loss_D_real = self.loss_D_real_B
        self.loss_D_fake = self.loss_D_fake_B
        if self.opt.decomp:
            self.loss_D_real +=  self.loss_D_real_B1 + self.loss_D_real_B2
            self.loss_D_fake += self.loss_D_fake_B1 + self.loss_D_fake_B2
            self.loss_D_real /= 3.0
            self.loss_D_fake /= 3.0

        #gradient penalty terms
        self.loss_gp_B = self.calc_gradient_penalty(self.netD_B, real_B.detach(), fake_B.detach())
        self.loss_gp_A1 = self.calc_gradient_penalty(self.netD_A1, real_B1, fake_B1.detach())
        self.loss_gp_A2 = self.calc_gradient_penalty(self.netD_A2, real_B2, fake_B2.detach())
        self.loss_gp = (self.loss_gp_B)
        if self.opt.decomp:
            self.loss_gp += self.loss_gp_A1 + self.loss_gp_A2
            self.loss_gp /= 3.0
        
        # Combined loss
        self.loss_D =  (0.5*(self.loss_D_fake + self.loss_D_real) + 
                        self.opt.lambda_gp* self.loss_gp)

        self.loss_D.backward()


    def backward_D_test(self):
        '''backward pass for the discriminator @ test time'''

        # Fake
        # stop backprop to the generator by detaching fake_B

        if self.opt.conditional:
            inp_D = self.fake_B
            fake_B1 = torch.cat((inp_D, self.fake_B1_T),1)
            fake_B2 = torch.cat((inp_D, self.fake_B2_T),1)
            #real
            inp_D = self.ex_B
            real_B1 = torch.cat((inp_D, self.ex_B1_T),1)
            real_B2 = torch.cat((inp_D, self.ex_B2_T),1)
        else:
            fake_B1 = self.fake_B1_T
            fake_B2 = self.fake_B2_T
            real_B1 = self.ex_B1_T
            real_B2 = self.ex_B2_T

        fake_BA1 = self.fake_BA1_pool.query(fake_B1.data)
        fake_BA2 = self.fake_BA2_pool.query(fake_B2.data)
        pred_fake_B1 = self.netD_A1(fake_BA1.detach())
        pred_fake_B2 = self.netD_A2(fake_BA2.detach())
        pred_real_B1 = self.netD_A1(real_B1)
        pred_real_B2 = self.netD_A2(real_B2)


        self.loss_D_fake_B1 = self.criterionGAN(pred_fake_B1,False)
        self.loss_D_real_B1 = self.criterionGAN(pred_real_B1,True)
        self.loss_D_fake_B2 = self.criterionGAN(pred_fake_B2,False)        
        self.loss_D_real_B2 = self.criterionGAN(pred_real_B2,True)


        if self.opt.conditional:
            fake_B = torch.cat((self.fake_A1_T, self.fake_A2_T, self.fake_B),1)
            real_B = torch.cat((self.ex_B1_T, self.ex_B2_T, self.ex_B),1)
        else:
            fake_B = self.fake_B
            real_B = self.ex_B

        pred_fake_B = self.netD_B(fake_B.detach())
        pred_real_B = self.netD_B(real_B.detach())

        self.loss_D_fake_B = self.criterionGAN(pred_fake_B,False)
        self.loss_D_real_B = self.criterionGAN(pred_real_B,True)
 
        self.loss_D_real = self.loss_D_real_B
        self.loss_D_fake = self.loss_D_fake_B
        if self.opt.decomp:
            self.loss_D_real += (self.loss_D_real_B1 + self.loss_D_real_B2 )
            self.loss_D_real /= 3.0
            self.loss_D_fake += (self.loss_D_fake_B1 + self.loss_D_fake_B2 )
            self.loss_D_fake /= 3.0

        #gradient penalty terms
        self.loss_gp_B1 = self.calc_gradient_penalty(self.netD_A1, real_B1.detach(), fake_B1.detach())
        self.loss_gp_B2 = self.calc_gradient_penalty(self.netD_A2, real_B2.detach(), fake_B2.detach())
        self.loss_gp_B = self.calc_gradient_penalty(self.netD_B, real_B.detach(), fake_B.detach())
        self.loss_gp = self.loss_gp_B
        if self.opt.decomp:
            self.loss_gp += (self.loss_gp_B1 + self.loss_gp_B2 )
            self.loss_gp /= 3.0

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real)*0.5 + self.opt.lambda_gp*self.loss_gp #+ self.loss_D_CLS

        self.loss_D.backward()


    def backward_G_test(self):
        '''backward pass for the generator @ test time'''

        lambda_A2 = torch.nonzero(self.fake_A1_T.data != 1).size(0) / torch.nonzero(self.fake_A2_T.data != 1).size(0)
        self.loss_G = 0

        # First, G(A1,A2) should fake the discriminator
        if self.opt.lambda_gan:
            if self.opt.conditional:
                inp_D = self.fake_B
                fake_B1 = torch.cat((inp_D.detach(), self.fake_B1_T),1)
                fake_B2 = torch.cat((inp_D.detach(), self.fake_B2_T),1)
            else:
                fake_B1 = self.fake_B1_T
                fake_B2 = self.fake_B2_T
            pred_fake_B1 = self.netD_A1(fake_B1)
            pred_fake_B2 = self.netD_A2(fake_B2)

            self.loss_G_GAN_B1 = self.criterionGAN(pred_fake_B1,True)
            self.loss_G_GAN_B2 = self.criterionGAN(pred_fake_B2,True)

            
            if self.opt.conditional:
                inp_pair = torch.cat((self.fake_A1_T, self.fake_A2_T),1)
                fake_B = torch.cat((inp_pair.detach(), self.fake_B),1)
            else:
                fake_B = self.fake_B
            pred_fake_B = self.netD_B(fake_B)

            self.loss_G_GAN_B = self.criterionGAN(pred_fake_B, True)

            self.loss_G_GAN  = self.loss_G_GAN_B 
            if self.opt.decomp:
                self.loss_G_GAN += (self.loss_G_GAN_B1 + self.loss_G_GAN_B2)/3.0

            self.loss_G += self.opt.lambda_gan * self.loss_G_GAN

        #pixel L1 loss
        self.loss_G_L1 = (0.5*(1.0/(1+lambda_A2) * self.criterionL1(self.fake_B1_T, self.fake_A1_T.detach()) +
                            float(lambda_A2)/(1+lambda_A2) * self.criterionL1(self.fake_B2_T, self.fake_A2_T.detach())) )

        #TODO delete
        n1 = (torch.mean(torch.sum(torch.sum(self.real_M1_s,dim=2),dim=2)))
        n2 = (torch.mean(torch.sum(torch.sum(self.real_M2_s,dim=2),dim=2)))
        n0 = (torch.mean(torch.sum(torch.sum(self.real_M0_s,dim=2),dim=2)))

        #TODO delete
        n_all = self.fake_B.size(2)*self.fake_B.size(3)
        # lambda_fg = n0.data[0] / (n_all)

        self.loss_G_seg = 0
        self.loss_G_seg +=  (0.5*self.criterionL1(torch.mul(self.real_M1_s.detach(), self.fake_B), 
                                    torch.mul(self.real_M1_s.detach(), self.fake_A1_T.detach()))+
                            0.5*self.criterionL1(torch.mul(self.real_M2_s.detach(), self.fake_B), 
                                    torch.mul(self.real_M2_s.detach(), self.fake_A2_T.detach())))
        # if there is a large background region:
        white_back = Variable(torch.ones(self.fake_B.size()).cuda())
        if n0.data[0]>0.2*self.fake_B.size(2)*self.fake_B.size(3):
            self.loss_G_seg += self.criterionL1(torch.mul(self.real_M0_s.detach(), self.fake_B),
                        torch.mul(self.real_M0_s.detach(), white_back))



        self.loss_G +=  self.opt.lambda_L2 * (self.loss_G_seg)
        if self.opt.decomp:
            self.loss_G += self.opt.lambda_L2 * self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters_completion(self, total_steps):
        self.forward_inpainting()
        D_freq = 10
        G_freq = 1

        self.optimizer_D1_completion.zero_grad()
        self.optimizer_D2_completion.zero_grad()
        self.backward_D_completion()
        if total_steps%D_freq== 0 :
            self.optimizer_D1_completion.step()
            self.optimizer_D2_completion.step()


        self.optimizer_G1_completion.zero_grad()
        self.optimizer_G2_completion.zero_grad()
        self.backward_G_completion()
        if total_steps%G_freq == 0:
            if self.opt.G1_completion:
                self.optimizer_G1_completion.step()
            if self.opt.G2_completion:
                self.optimizer_G2_completion.step()

    def optimize_parameters_STN(self):

        self.forward_STN()
        self.optimizer_STN_dec.zero_grad()
        self.optimizer_STN_c.zero_grad()
        self.backward_STN()
        self.optimizer_STN_c.step()
        self.optimizer_STN_dec.step()

    def optimize_parameters(self, total_steps):
        if self.opt.G1_completion or self.opt.G2_completion:
            self.optimize_parameters_completion(total_steps)
        self.forward()
        D_freq = 10
        G_freq = 1

        self.optimizer_D_A1.zero_grad()
        self.optimizer_D_A2.zero_grad()
        self.optimizer_D_B.zero_grad()

        if self.opt.lambda_gan:
            self.backward_D('A')
            if total_steps%D_freq== 0 :
                self.optimizer_D_A1.step()
                self.optimizer_D_A2.step()
                self.optimizer_D_B.step()

        self.optimizer_G_comp.zero_grad()
        self.optimizer_G_decomp.zero_grad()
        self.optimizer_STN_c.zero_grad()

        self.backward_G()
        if total_steps%G_freq == 0:
            self.optimizer_G_decomp.step()
            self.optimizer_G_comp.step()
            self.optimizer_STN_c.step()


    def optimize_parameters_test(self, total_steps):
        self.forward_test()
        D_freq = 10
        G_freq = 1

        self.optimizer_D_A1.zero_grad()
        self.optimizer_D_A2.zero_grad()
        self.optimizer_D_B.zero_grad()
        self.backward_D_test()
        if total_steps%D_freq== 0 :
            self.optimizer_D_A1.step()
            self.optimizer_D_A2.step()
            self.optimizer_D_B.step()

        self.optimizer_G_comp.zero_grad()
        self.optimizer_G_decomp.zero_grad()
        self.backward_G_test()
        if total_steps%G_freq == 0:
            self.optimizer_G_comp.step()


    def optimize_parameters_test_random(self, index, theta, total_steps,rand_inds):
        self.forward_test_random_STN(index, theta,rand_inds)
        D_freq = 10
        G_freq = 1

        self.optimizer_D_A1.zero_grad()
        self.optimizer_D_A2.zero_grad()
        self.optimizer_D_B.zero_grad()
        self.backward_D_test()
        if total_steps%D_freq== 0 :
            self.optimizer_D_A1.step()
            self.optimizer_D_A2.step()
            self.optimizer_D_B.step()

        self.optimizer_G_comp.zero_grad()
        self.optimizer_G_decomp.zero_grad()
        self.backward_G_test()
        if total_steps%G_freq == 0:
            self.optimizer_G_comp.step()


    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                            ('D_real', self.loss_D_real.data[0]),
                            ('D_fake', self.loss_D_fake.data[0]),
                            ('G_L1', self.loss_G_L1.data[0]),
                            ('GP', self.loss_gp.data[0]),
                            ('G_seg', self.loss_G_seg.data[0]),
                            ('G_AFN', self.loss_AFN.data[0]),
                            ('G_mask', self.loss_G_mask.data[0]),
                            ('STN', self.loss_STN.data[0]),
                            ('G_compl', self.loss_G_completion.data[0]),
                            ('D_compl', self.loss_D_completion.data[0])
                            ])

    def get_current_visuals_STN(self):
        vis_tensors = ['stn_B1_T', 'stn_B2_T', 'stn_B1', 'stn_B2','real_B', 'real_B1_T', 
                    'real_B2_T', 'real_B1', 'real_B2']

        vis_ims = self.tnsrs2ims(vis_tensors)

        visuals = zip(vis_tensors, vis_ims)
        return OrderedDict(visuals)

    def get_current_visuals_completion(self):
        vis_tensors = ['segment_A1', 'segment_A2', 'real_A1_T',
                        'real_A2_T', 'fake_A1_compl', 'fake_A2_compl']

        vis_ims = self.tnsrs2ims(vis_tensors)
        visuals = zip(vis_tensors, vis_ims)
        return OrderedDict(visuals)


    def get_current_visuals_A_segment(self):


        vis_tensors = ['real_A1', 'fake_A1', 'real_A2', 'fake_A2'
                        ,'real_A1_T','fake_A1_T', 'real_A2_T','fake_A2_T'
                        , 'real_B', 'fake_B', 'fake_B_sum','real_M','fake_M']
        if self.opt.lambda_mask:
            self.real_M = 86*self.real_M1_s + 172*self.real_M2_s
            self.fake_M = 86*self.fake_M1_s + 172*self.fake_M2_s
        vis_tensors += ['real_M','fake_M']


        vis_ims = self.tnsrs2ims(vis_tensors)
        visuals = zip(vis_tensors, vis_ims)
                            
        return OrderedDict(visuals)


    def get_current_visuals(self):
        vis_tensors = ['real_A1', 'fake_A1', 'real_A2', 'fake_A2'
                ,'fake_A1_T','fake_A2_T'
                , 'fake_B', 'fake_B_sum']
        if self.opt.random_view:
            vis_tensors += ['real_A1_r']

        if self.opt.lambda_mask:
            self.fake_M = 86*self.fake_M1_s + 172*self.fake_M2_s
            self.real_M = 86*self.real_M1_s + 172*self.real_M2_s
        vis_tensors += ['fake_M','real_M']

        vis_ims = self.tnsrs2ims(vis_tensors)
        visuals = zip(vis_tensors, vis_ims)
                            
        return OrderedDict(visuals)

    def save(self, label, compl_pretrain=False, STN_pretrain=False):
        if compl_pretrain:
            model_names = ['G1_completion','D1_completion','G2_completion','D2_completion']
            labels = [label]*4
        
        elif STN_pretrain:
            model_names = ['STN_c','STN_dec']
            labels = [label]*2

        else:
            model_names =['G_comp', 'G_decomp','D_A1', 'D_A2', 'D_B']
            labels = [label]*5
            model_names += ['STN_dec', 'STN_c']
            if label=='latest':
                labels += [label]*2
            else:
                labels += [str(int(label)+int(self.opt.which_epoch_STN))]*2

            if self.opt.img_completion:
                try:
                    compl_label = str(int(label)+int(self.opt.which_epoch_completion))
                except:
                    compl_label = label
                model_names += ['G1_completion','D1_completion','G2_completion','D2_completion']
                labels += [compl_label]*4

            if self.opt.random_view:
                if label=='latest':
                    AFN_label = label
                else:
                    AFN_label = str(int(label)+int(self.opt.which_epoch_AFN))
                model_names += ['AFN']
                labels += [AFN_label]
        self.save_networks(model_names, labels)




