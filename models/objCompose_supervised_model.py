# ========================================================
# Compositional GAN
# Model for the Paired Training Data
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
from scipy import misc
from torch import index_select, LongTensor, nn
import torch.nn.functional as F


class objComposeSuperviseModel(BaseModel):
    def name(self):
        return 'objComposeSuperviseModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.y_x = int(float(opt.fineSizeY)/opt.fineSizeX)

        # -------------------------------------
        # Define Networks
        # -------------------------------------
        # Composition Generator
        self.netG_comp = networks.define_G(2*opt.output_nc, opt.input_nc, opt.ngf,opt.which_model_netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      opt.noise, y_x=self.y_x)
        
        if opt.lambda_mask:                        
            opt.which_model_netG = '%s_masked'%opt.which_model_netG

        #Decomposition Generator
        self.netG_decomp = networks.define_G(opt.input_nc, 2*opt.output_nc, opt.ngf,opt.which_model_netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,opt.noise,
                                      y_x=self.y_x)
                                      
        if opt.random_view:
            # Relative Appearance Flow Network
            self.netAFN = networks.define_AFN(opt.which_model_AFN, input_nc=opt.input_nc+1,init_type=opt.init_type, gpu_ids=self.gpu_ids)
            if self.isTrain and not opt.continue_train:
                self.load_network(self.netAFN, 'AFN', opt.which_epoch_AFN)
            else:
                self.load_network(self.netAFN, 'AFN', int(opt.which_epoch_AFN)+int(opt.which_epoch))

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
            self.netD_A1 = networks.define_D(inp_disc, opt.ndf,opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,y_x=self.y_x)
        
            self.netD_A2 = networks.define_D(inp_disc, opt.ndf,opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,y_x=self.y_x)            

            if opt.conditional:
                in_ch = opt.input_nc*3
            else:
                in_ch = opt.input_nc
            self.netD_B = networks.define_D(in_ch, opt.ndf,opt.which_model_netD,opt.n_layers_D,
                                         opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, y_x=self.y_x)

        
        # ---------------------------------
        # Load networks
        #----------------------------------

        if not self.isTrain or opt.continue_train:
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
            self.load_networks()



        # ---------------------------------
        # initialize optimizers
        # ---------------------------------
        self.model_names = ['G_decomp','G_comp','STN_dec','STN_c']
        if self.isTrain:
            self.model_names += ['D_A1','D_A2','D_B']
        if opt.random_view:
            self.model_names += ['AFN']

        if self.isTrain:
            self.schedulers = []
            self.set_optimizers()
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))
            self.print_networks(verbose=False)

            self.fake_BA1_pool = ImagePool(opt.pool_size)
            self.fake_BA2_pool = ImagePool(opt.pool_size)

            # ----------------------------------
            # define loss functions
            # ----------------------------------
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionCLS = nn.CrossEntropyLoss()
            self.criterionbCLS = nn.BCELoss()
            self.L2_loss = torch.nn.MSELoss()

            # loss function initializations
            self.loss_G_GAN = Variable(torch.Tensor(1).fill_(0).cuda())
            self.loss_D_real = Variable(torch.Tensor(1).fill_(0).cuda())
            self.loss_D_fake = Variable(torch.Tensor(1).fill_(0).cuda())
            self.loss_G_L1 = Variable(torch.Tensor(1).fill_(0).cuda())
            self.loss_gp = Variable(torch.Tensor(1).fill_(0).cuda())
            self.loss_AFN = Variable(torch.Tensor(1).fill_(0).cuda())
            self.loss_STN = Variable(torch.Tensor(1).fill_(0).cuda())
            self.loss_segmetation = Variable(torch.Tensor(1).fill_(0).cuda())
        self.softmax = torch.nn.Softmax(dim=1)
   


    def set_input_train(self, input):
        ''' samples of real distribution (from training set) to be used at test time'''
        self.ex_B = input['B']

        # full center-oriented input objects B1,B2 
        # (and their corresponding transposed images) mapped with the composite image B
        self.ex_B1 = input['B1']
        self.ex_B2 = input['B2']
        self.ex_B1_T = input['B1_T'] 
        self.ex_B2_T = input['B2_T']
        input_vars = ['ex_B', 'ex_B1', 'ex_B2',
                      'ex_B1_T', 'ex_B2_T']
        if len(self.gpu_ids) > 0:
            self.tocuda(input_vars)
        for name in input_vars:
            if isinstance(name, str):
                var = getattr(self, name)
                setattr(self, name, Variable(var))

    def set_input_test(self, input):
        ''' samples at test time; no target composite image given'''

        self.input_A1 = input['A1']
        self.input_A2 = input['A2']
        self.A_paths = input['A_paths']

        input_vars = ['input_A1', 'input_A2']

        if len(self.gpu_ids) > 0:
            self.tocuda(input_vars)

        self.real_A1 = Variable(self.input_A1)
        self.real_A2 = Variable(self.input_A2)



    def set_input(self, input):
        '''Samples at training time'''

        self.input_A1 = input['A1']
        self.input_A2 = input['A2']
        self.input_B = input['B']
        self.input_B1 = input['B1']
        self.input_B2 = input['B2']
        self.input_B1_T = input['B1_T']
        self.input_B2_T = input['B2_T']
        self.A_paths = input['A_paths']
        self.B_paths = input['B_paths']

        if len(self.gpu_ids) > 0:
            self.input_A1 = self.input_A1.cuda(self.gpu_ids[0], async=True)
            self.input_A2 = self.input_A2.cuda(self.gpu_ids[0], async=True)
            self.input_B1 = self.input_B1.cuda(self.gpu_ids[0], async=True)
            self.input_B2 = self.input_B2.cuda(self.gpu_ids[0], async=True)
            self.input_B1_T = self.input_B1_T.cuda(self.gpu_ids[0], async=True)
            self.input_B2_T = self.input_B2_T.cuda(self.gpu_ids[0], async=True)
            self.input_B = self.input_B.cuda(self.gpu_ids[0], async=True)


        #ground truth segmentation masks
        #TODO: change to input masks
        diff_0 = torch.sum(torch.pow(self.input_B-1, 2),dim=1, keepdim=True)
        diff_1 = torch.sum(torch.pow(self.input_B1_T - self.input_B, 2),dim=1, keepdim=True)
        diff_2 = torch.sum(torch.pow(self.input_B2_T - self.input_B, 2),dim=1, keepdim=True)
        val, self.real_M = torch.min(torch.cat((diff_0, diff_1, diff_2), 1),1)
        self.real_M1_s = Variable((self.real_M==1).unsqueeze(1).type(torch.cuda.FloatTensor))
        self.real_M2_s = Variable((self.real_M==2).unsqueeze(1).type(torch.cuda.FloatTensor))
        self.real_M = Variable(self.real_M.type(torch.cuda.LongTensor))


        self.real_A1 = Variable(self.input_A1)
        self.real_A2 = Variable(self.input_A2)
        self.real_B1 = Variable(self.input_B1)
        self.real_B2 = Variable(self.input_B2)
        self.real_B1_T = Variable(self.input_B1_T)
        self.real_B2_T = Variable(self.input_B2_T)
        self.real_B = Variable(self.input_B) 

    def forward(self):
        '''starting from input object images'''

        self.mask_A2 = (torch.mean(self.real_A2,dim=1,keepdim=True)<1).type(torch.cuda.FloatTensor) 
        self.mask_A1 = (torch.mean(self.real_A1,dim=1,keepdim=True)<1).type(torch.cuda.FloatTensor).repeat(1,3,1,1) 

        #start the cycle from real_A1, real_A2
        #-------------------------
        #AFN
        #-------------------------
        if self.opt.random_view:
            self.flow, self.mask_pred = self.netAFN(torch.cat((self.real_A1, self.mask_A2),1))
            self.flow = self.flow.permute(0,2,3,1)
            # grid values should be in the range [-1,1]
            self.fake_A1 = F.grid_sample(self.real_A1, self.flow)
        else:
            self.fake_A1 = self.real_A1
        self.fake_A2 = self.real_A2

        #-------------------------
        # Composition network
        #-------------------------
        self.fake_A1_T, self.fake_A2_T = (self.netSTN_c(torch.cat((self.fake_A1.detach(),self.fake_A2),1)))
        self.fake_A = torch.cat((self.fake_A1_T,self.fake_A2_T),1)
        self.fake_B = self.netG_comp(self.fake_A)

        #-------------------------
        # Decomposition networks
        #-------------------------
        self.B1_B2, self.M1_M2 = self.netG_decomp(self.fake_B)
        self.M1_M2_normal = self.softmax(self.M1_M2)
        v,m = torch.max(self.M1_M2_normal, dim=1, keepdim=True)
        self.fake_M1_s = ((m==1)*1).type(torch.cuda.FloatTensor)
        self.fake_M2_s = ((m==2)*1).type(torch.cuda.FloatTensor)
        if self.opt.phase=='test':
            self.M1_M2 = self.softmax(self.M1_M2)


        self.fake_B.retain_grad()
        self.fake_B1_T = index_select(self.B1_B2,1,Variable(LongTensor(range(0,self.opt.input_nc)).cuda()))
        self.fake_B2_T = index_select(self.B1_B2,1,Variable(LongTensor(range(self.opt.input_nc,2*self.opt.input_nc)).cuda()))
        self.fake_B1,self.fake_B2 = self.netSTN_dec(torch.cat((self.fake_B1_T, self.fake_B2_T),1))

        #TODO
        self.mask_A1_fake = torch.mean(self.fake_A1_T,dim=1,keepdim=True)
        self.mask_A2_fake = torch.mean(self.fake_A2_T,dim=1,keepdim=True)

        self.mask_A1_fake = (self.mask_A1_fake<self.opt.Thresh1*torch.max(self.fake_A1_T).data).type(torch.cuda.FloatTensor)
        self.mask_A2_fake = (self.mask_A2_fake<self.opt.Thresh2*torch.max(self.fake_A2_T).data).type(torch.cuda.FloatTensor)

        self.mask_A0_fake = 1 - (((self.mask_A1_fake + self.mask_A2_fake)>=1)*1).type(torch.cuda.FloatTensor)
        # AND between object mask and predicted mask
        self.fake_M1_s_ = torch.mul(self.fake_M1_s, self.mask_A1_fake).repeat(1,3,1,1)
        self.fake_M2_s_ = torch.mul(self.fake_M2_s, self.mask_A2_fake).repeat(1,3,1,1)

        forgot_overlp = (((self.fake_M1_s_ + self.fake_M2_s_)==0)*1).type(torch.cuda.FloatTensor)

        self.forgot_A1 = torch.mul(forgot_overlp, self.mask_A1_fake)
        self.forgot_A2 = torch.mul(forgot_overlp, self.mask_A2_fake)

        self.fake_M1_s_ += self.forgot_A1
        self.fake_M2_s_ += self.forgot_A2
        self.fake_B_sum = (torch.mul(self.fake_M1_s_, self.fake_A1_T) + torch.mul(self.fake_M2_s_, self.fake_A2_T)
                            + (1-self.fake_M1_s_ - self.fake_M2_s_))


    def forward_test(self):
        '''starting from input object images @ test time'''

        self.forward()

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


    def backward_D(self):
        '''backward pass for the discriminator in training the paired model'''

        # Fake
        # stop backprop to the generator by detaching fake_B
        if self.opt.conditional:
            inp_D = self.fake_B
            
            fake_B1 = torch.cat((inp_D, self.fake_B1_T),1)
            fake_B2 = torch.cat((inp_D, self.fake_B2_T),1)
            real_B1 = torch.cat((inp_D.detach(), self.real_B1_T),1)
            real_B2 = torch.cat((inp_D.detach(), self.real_B2_T),1)
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
            fake_B = torch.cat((self.fake_A1_T, self.fake_A2_T, self.fake_B),1)
            real_B = torch.cat((self.fake_A1_T, self.fake_A2_T, self.real_B),1)
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


    def backward_G(self):
        '''backward pass for the generator in training the paired model'''
        lambda_A2 = torch.nonzero(self.fake_A1_T.data != 1).size(0) / torch.nonzero(self.fake_A2_T.data != 1).size(0)
        self.loss_G = 0 

        #--------------------------
        #GAN Loss
        #--------------------------
        # First, G(A1,A2) should fake the discriminator
        if self.opt.lambda_gan:
            if self.opt.conditional:
                inp_D = self.fake_B
                pred_fake_B1 = self.netD_A1(torch.cat((inp_D.detach(), self.fake_B1_T),1))
                pred_fake_B2 = self.netD_A2(torch.cat((inp_D.detach(), self.fake_B2_T),1))
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

            self.loss_G_GAN  = self.loss_G_GAN_B
            if self.opt.decomp:
                self.loss_G_GAN += (self.loss_G_GAN_B1 + self.loss_G_GAN_B2)
                self.loss_G_GAN /= 3.0

            self.loss_G +=  self.opt.lambda_gan * self.loss_G_GAN
        #--------------------------
        # pixel L1 loss
        #--------------------------
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        L2_loss = torch.nn.MSELoss()

        if self.opt.decomp:
            self.loss_G_L1 += 0.5*(self.criterionL1(self.fake_B1, self.fake_A1.detach()) +
                               self.criterionL1(self.fake_B2, self.fake_A2.detach()))
        
        
            self.loss_G_L1 += ( 1.0/(1+lambda_A2) * L2_loss(self.fake_B1_T, self.real_B1_T)+
                    lambda_A2/(1+lambda_A2) * L2_loss(self.fake_B2_T, self.real_B2_T))

        self.loss_G_L1 += (1.0/(1+lambda_A2) * L2_loss(self.fake_A1_T,self.real_B1_T)+
                           lambda_A2/(1+lambda_A2) *L2_loss(self.fake_A2_T,self.real_B2_T))



        self.loss_G += self.opt.lambda_L2 * self.loss_G_L1 

        #--------------------------
        # Loss for training RAFN
        #--------------------------
        self.loss_AFN = Variable(torch.Tensor(1).fill_(0).cuda())
        if self.opt.random_view:
            if self.opt.lambda_AFN:
                self.loss_G_AFN = 0
                self.loss_G_AFN += self.criterionL1(self.fake_A1, self.real_B1)

                self.loss_G_AFN += 0.1*self.criterionL1(torch.mul(self.fake_A1, self.mask_A1),
                                                     torch.mul(self.real_B1, self.mask_A1))
            
                self.mask_tgt = index_select(self.mask_A1, 1, Variable(LongTensor([1])).cuda())

                self.loss_binary = self.criterionbCLS(self.mask_pred, self.mask_tgt)

                self.loss_AFN = self.loss_G_AFN + 0.5*self.loss_binary
                self.loss_G += self.opt.lambda_AFN * self.loss_AFN

        #--------------------------
        # cross entropy loss for the mask prediction network
        #--------------------------
        if self.opt.lambda_mask:
            self.loss_segmetation = self.opt.lambda_mask * self.criterionCLS(self.M1_M2, self.real_M)
            self.loss_G += self.loss_segmetation


        self.loss_G.backward()



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

        self.loss_gp_B1 = self.calc_gradient_penalty(self.netD_A1, real_B1.detach(), fake_B1.detach())
        self.loss_gp_B2 = self.calc_gradient_penalty(self.netD_A2, real_B2.detach(), fake_B2.detach())
        self.loss_gp_B = self.calc_gradient_penalty(self.netD_B, real_B.detach(), fake_B.detach())
        self.loss_gp = self.loss_gp_B
        if self.opt.decomp:
            self.loss_gp += (self.loss_gp_B1 + self.loss_gp_B2 )
            self.loss_gp /= 3.0

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real)*0.5 + self.opt.lambda_gp*self.loss_gp

        self.loss_D.backward()


    def backward_G_test(self):
        '''backward pass for the generator @ test time'''

        lambda_A2 = torch.nonzero(self.fake_A1_T.data != 1).size(0) / torch.nonzero(self.fake_A2_T.data != 1).size(0)
        self.loss_G = 0
        #-----------------------
        #GAN loss
        #-----------------------        
        # First, G(A) should fake the discriminator
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
                self.loss_G_GAN += (self.loss_G_GAN_B1 + self.loss_G_GAN_B2)
                self.loss_G_GAN /= 3.0

            self.loss_G += self.opt.lambda_gan * self.loss_G_GAN

        #------------------------------
        # pixel L1 loss
        #------------------------------
        # self-consistent loss
        self.loss_G_L1 = 0
        if self.opt.decomp:
            self.loss_G_L1 = (0.5*(1.0/(1+lambda_A2) * self.criterionL1(self.fake_B1_T, self.fake_A1_T.detach()) +
                                  lambda_A2/(1+lambda_A2) * self.criterionL1(self.fake_B2_T, self.fake_A2_T.detach())) )
            self.loss_G += self.opt.lambda_L2 * self.loss_G_L1

        n0 = (torch.mean(torch.sum(torch.sum(self.mask_A0_fake,dim=2),dim=2))) #number of background pixels

        # fake_B should follow the style and color of the inputs
        self.loss_masks = 0        
        white_back = Variable(torch.ones(self.fake_B.size()).cuda())
        self.loss_masks += (1.0/2)*(self.criterionL1(torch.mul(self.fake_M1_s_.detach(), self.fake_B), 
                                    torch.mul(self.fake_M1_s_.detach(), self.fake_A1_T.detach()))+
                                self.criterionL1(torch.mul(self.fake_M2_s_.detach(), self.fake_B), 
                                    torch.mul(self.fake_M2_s_.detach(), self.fake_A2_T.detach())))#+

        if n0.data>0.3*self.fake_B.size(2)*self.fake_B.size(3):
            # if there is a large background area apply an L1 loss on the background region too
            self.loss_masks +=self.criterionL1(torch.mul(self.mask_A0_fake.detach(), self.fake_B),
                        torch.mul(self.mask_A0_fake.detach(), white_back))

        self.loss_G +=  self.opt.lambda_L2 * (self.loss_masks)
        

        self.loss_G.backward()



    def optimize_parameters(self, total_steps, epoch):
        if (int(total_steps/10))%1 == 0:
            self.forward()
        D_freq = 10
        G_freq = 1

        self.optimizer_D_A1.zero_grad()
        self.optimizer_D_A2.zero_grad()
        self.optimizer_D_B.zero_grad()
        if self.opt.random_view:
            self.optimizer_AFN.zero_grad()

        if self.opt.lambda_gan:
            self.backward_D()
            if total_steps%D_freq== 0 :
                self.optimizer_D_A1.step()
                self.optimizer_D_A2.step()
                self.optimizer_D_B.step()

        self.optimizer_STN_dec.zero_grad()
        self.optimizer_STN_c.zero_grad()

        self.optimizer_G_comp.zero_grad()
        self.optimizer_G_decomp.zero_grad()
        self.backward_G()
        if total_steps%G_freq == 0:
            self.optimizer_STN_dec.step()
            self.optimizer_G_decomp.step()
            self.optimizer_G_comp.step()
            self.optimizer_STN_c.step()
            if self.opt.random_view:
                self.optimizer_AFN.step()


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


    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data),
                            ('D_real', self.loss_D_real.data),
                            ('D_fake', self.loss_D_fake.data),
                            ('G_L1', self.loss_G_L1.data),
                            ('GP', self.loss_gp.data),
                            ('G_AFN', self.loss_AFN.data),
                            ('G_mask', self.loss_segmetation.data)
                            ])

    def get_current_errors_test(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data),
                            ('D_real', self.loss_D_real.data),
                            ('D_fake', self.loss_D_fake.data),
                            ('G_L1', self.loss_G_L1.data),
                            ('GP', self.loss_gp.data)
                            ])


    def get_current_visuals(self):
        vis_tensors = ['real_A1', 'real_A2', 'fake_A1','fake_A2',
                        'fake_A1_T', 'fake_A2_T','fake_B1', 'fake_B2',
                        'fake_B', 'fake_B_sum']
        if self.opt.phase != 'test':               
            vis_tensors += ['real_B']
        if self.opt.lambda_mask:
            if self.opt.phase != 'test':
                self.real_M_vis = 86*self.real_M1_s + 172*self.real_M2_s
                vis_tensors += ['real_M_vis']
            self.fake_M_vis = 86*self.fake_M1_s + 172*self.fake_M2_s
            vis_tensors += ['fake_M_vis']

        vis_ims = self.tnsrs2ims(vis_tensors)
        visuals = zip(vis_tensors, vis_ims)

        return OrderedDict(visuals)

    def save(self, label, cls_pretrain=False):
        model_names = ['G_comp', 'G_decomp', 'D_A1', 'D_A2', 'D_B','STN_dec', 'STN_c']
        labels = [label] * 7
        if self.opt.random_view:
            if label=='latest':
                AFN_label = label
            else:
                AFN_label = str(int(label)+int(self.opt.which_epoch_AFN))
            model_names += ['AFN']
            labels += [AFN_label]
        self.save_networks(model_names, labels)




