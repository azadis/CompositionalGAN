import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
from torch import index_select, LongTensor, nn
import torch.nn.functional as F

'''Relative AFN model given an object and the binary mask of the second object as the viewpoint reference'''
class AFNComposeModel(BaseModel):
    def name(self):
        return 'AFNComposeModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define networks
        self.netAFN = networks.define_AFN(opt.AFNmodel, input_nc=opt.input_nc+1,init_type=opt.init_type, gpu_ids=self.gpu_ids)


        # loss function initializations
        self.loss_G_AFN = Variable(torch.Tensor(1).fill_(0).cuda())
        self.loss_binary = Variable(torch.Tensor(1).fill_(0).cuda())
        self.loss_G = Variable(torch.Tensor(1).fill_(0).cuda())
        self.loss_perceptual = Variable(torch.Tensor(1).fill_(0).cuda())
        self.sigmoid = nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netAFN, 'AFN', opt.which_epoch)

        self.softmax =  nn.Softmax(dim=1)
        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionCLS = nn.BCELoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_AFN = torch.optim.Adam(self.netAFN.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_AFN)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netAFN)
        print('-----------------------------------------------')



    def set_input(self, input):
        self.input_A1 = input['A1']
        self.input_A2 = input['A2']
        self.A_paths = input['A_paths']

        self.input_B = input['B']
        self.B_paths = input['B_paths']

        if len(self.gpu_ids) > 0:
            self.input_A1 = self.input_A1.cuda(self.gpu_ids[0], async=True)
            self.input_A2 = self.input_A2.cuda(self.gpu_ids[0], async=True)
            self.input_B = self.input_B.cuda(self.gpu_ids[0], async=True)



    def forward_AFN(self):

        self.real_A1 = Variable(self.input_A1)
        self.real_A2 = Variable(self.input_A2)
        self.real_B = Variable(self.input_B)
        self.mask_A1 = (torch.mean(self.real_B,dim=1,keepdim=True)<1).type(torch.cuda.FloatTensor).repeat(1,3,1,1)

        self.flow, self.mask_pred = self.netAFN(torch.cat((self.real_A1, self.real_A2),1))
        self.flow = self.flow.permute(0,2,3,1)
        # grid values should be in the range [-1,1]
        self.fake_A1 = F.grid_sample(self.real_A1, self.flow)
        if self.opt.phase=='test':
            self.mask_pred = (self.mask_pred > 0.5).float() * 1

        self.mask_tgt = index_select(self.mask_A1, 1, Variable(LongTensor([1])).cuda())


    # get image paths
    def get_image_paths(self):
        if self.opt.phase == 'test':
            return self.A_paths[0]
        else:
            return self.B_paths

    def backward_AFN(self):
        self.loss_G = self.criterionL1(self.fake_A1, self.real_B)
        self.loss_G += 0.1*self.criterionL1(torch.mul(self.fake_A1, self.mask_A1),
                                             torch.mul(self.real_B, self.mask_A1))        

        self.loss_binary = self.criterionCLS(self.mask_pred, self.mask_tgt)

        self.loss_F = self.loss_G + 0.5*self.loss_binary

        self.loss_F.backward()




    def optimize_parameters_AFN(self):
        self.forward_AFN()
        self.optimizer_AFN.zero_grad()

        self.backward_AFN()
        self.optimizer_AFN.step()

    def get_current_errors(self):

        return OrderedDict([('G', self.loss_G.data[0]),
                            ('F', self.loss_F.data[0]),
                            ('binary', self.loss_binary.data[0])
                            ])


    def get_current_visuals(self):
        input_A1 = util.tensor2im(self.input_A1)
        input_A2 = util.tensor2im(self.input_A2)
        real_B = util.tensor2im(self.input_B)
        fake_A1 = util.tensor2im(self.fake_A1.data)
        mask_pred = util.tensor2im(self.mask_pred.data)
        mask_tgt = util.tensor2im(self.mask_tgt.data)

        visuals = [('input_A1', input_A1),('input_A2', input_A2)
                    ,('real_B', real_B)
                    ,('fake_A1', fake_A1)
                    ,('mask_pred',mask_pred), ('mask_tgt',mask_tgt)
                ]
        return OrderedDict(visuals)

    def save(self, label, cls_pretrain=False):
        self.save_network(self.netAFN, 'AFN', label, self.gpu_ids)
