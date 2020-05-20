import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
from torch import index_select, LongTensor, nn
import torch.nn.functional as F
import torchvision.models as models

class AFNModel(BaseModel):
    def name(self):
        return 'AFNModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define networks

        self.netAFN = networks.define_AFN(self.opt.AFNmodel, flow_mask=0, res=opt.fineSize, init_type=opt.init_type, use_dropout=not opt.no_dropout, gpu_ids=self.gpu_ids)
        
        # loss function initializations
        self.loss_binary = Variable(torch.Tensor(1).fill_(0).cuda())
        self.loss_G = Variable(torch.Tensor(1).fill_(0).cuda())
        self.loss_perceptual = Variable(torch.Tensor(1).fill_(0).cuda())
        self.sigmoid = nn.Sigmoid()

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netAFN, 'AFN', opt.which_epoch)

        self.softmax =  nn.Softmax(dim=1)
        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionCLS = nn.BCELoss()
            self.criterionL2 = nn.MSELoss()
            self.cross_entropy = nn.CrossEntropyLoss()

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
        self.input = input['A']
        self.target = input['B']
        self.A_paths = input['A_paths']
        self.mask_inp = input['mask_A']
        self.mask_tgt = input['mask_B']
        self.view_trans = input['trans']

        self.A_paths = input['A_paths']
        self.B_paths = input['B_paths']

        if len(self.gpu_ids) > 0:
            self.input = self.input.cuda(self.gpu_ids[0], async=True)
            self.target = self.target.cuda(self.gpu_ids[0], async=True)
            self.mask_inp = self.mask_inp.cuda(self.gpu_ids[0], async=True)
            self.mask_tgt = self.mask_tgt.cuda(self.gpu_ids[0], async=True)
            self.view_trans = self.view_trans.cuda(self.gpu_ids[0], async=True)



    def flow_to_grid(self, flow):
        gx, gy = np.meshgrid(range(0, flow.size(2)), \
                            range(0, flow.size(3)))
        gx = (gx/flow.size(2)-0.5)*2
        gy = (gy/flow.size(3)-0.5)*2
        gx = torch.Tensor(gx)
        gy = torch.Tensor(gy)
        if len(self.gpu_ids) > 0:
            gx = gx.cuda(self.gpu_ids[0], async=True)
            gy = gy.cuda(self.gpu_ids[0], async=True)

        gxy = Variable(torch.cat((gx.unsqueeze(0), gy.unsqueeze(0)), 0)).unsqueeze(0)
        gxy = gxy.repeat(flow.size(0), 1, 1, 1)
        flow = (flow + gxy)/2.0
        flow = torch.clamp(flow, -1, 1)
        flow = flow.permute(0,2,3,1)
        return flow


    def forward(self):

        self.input = Variable(self.input)

        self.target = Variable(self.target)
        self.view_trans = Variable(self.view_trans)

        self.flow,self.mask_pred = self.netAFN(self.input, self.view_trans)
        self.flow = self.flow.permute(0,2,3,1)
        # self.flow = self.flow_to_grid(self.netAFN(self.input, self.view_trans))
        # # grid values should be in the range [-1,1]
        self.output = F.grid_sample(self.input, self.flow)


        self.mask_tgt = Variable(self.mask_tgt)

    
    # no backprop gradients
    def test(self):

        self.input = Variable(self.input)
        self.target = Variable(self.target)
        self.mask_tgt = Variable(self.mask_tgt)
        self.view_trans = Variable(self.view_trans)
        self.netAFN.eval()

        self.flow,self.mask_pred = self.netAFN(self.input, self.view_trans)
        self.netAFN.train()
        self.flow = self.flow.permute(0,2,3,1)
        # grid values should be in the range [-1,1]
        self.output = F.grid_sample(self.input, self.flow)

        #taking argmax 
        self.mask_pred = (self.mask_pred > 0.5).float() * 1

    # get image paths
    def get_image_paths(self):
        if self.opt.phase == 'test':
            return [self.A_paths[0]]
        else:
            return self.B_paths
    

    def backward_flow(self):

        self.loss_G = self.criterionL1(self.output, self.target)
        if self.opt.lambda_masked:
            self.loss_G += self.opt.lambda_masked*self.criterionL1(torch.mul(self.output, self.mask_tgt),
                                                 torch.mul(self.target, self.mask_tgt))
        
        mask_tgt = index_select(self.mask_tgt, 1, Variable(LongTensor([1])).cuda())
        self.loss_binary = self.criterionCLS(self.mask_pred, mask_tgt)

        self.loss_F = self.loss_G + 0.5*self.loss_binary
        self.loss_F.backward()

    def backward_mask(self):
        mask_tgt = index_select(self.mask_tgt, 1, Variable(LongTensor([1])).cuda()).squeeze(1).type(torch.cuda.LongTensor)
        self.loss_binary = self.cross_entropy(self.mask_pred, mask_tgt)
        self.loss_binary.backward()



    def optimize_parameters(self):
        self.forward()
        self.optimizer_AFN.zero_grad()

        self.backward_flow()
        self.optimizer_AFN.step()

    def get_current_errors(self):
        return OrderedDict([('G', self.loss_G.data[0]),
                            ('F', self.loss_F.data[0]),
                            ('binary', self.loss_binary.data[0])
                            ])


    def get_current_visuals(self):
        input_ = util.tensor2im(self.input.data)
        target = util.tensor2im(self.target.data)
        output = util.tensor2im(self.output.data)
        mask = util.tensor2im(self.mask_pred.data)
        mask_tgt = util.tensor2im(self.mask_tgt.data)
        visuals = [('input', input_),('output', output),\
                    ('target', target), ('mask', mask),\
                    ('mask_tgt', mask_tgt)]

        return OrderedDict(visuals)

    def save(self, label, cls_pretrain=False):
        self.save_network(self.netAFN, 'AFN', label, self.gpu_ids)




