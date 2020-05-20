# ========================================================
# Compositional GAN
# Network architectures
# By Samaneh Azadi
# ========================================================


import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
from torch import index_select, LongTensor, nn
from scipy import misc

###############################################################################
# Functions
###############################################################################

def weights_init_constant(m):
    classname = m.__class__.__name__
    # print("classname:",classname)
    if classname.find('Conv') != -1:
        init.constant_(m.weight.data, 0.0)
    elif classname.find('Linear') != -1:
        init.constant_(m.weight.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 0.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print("classname:",classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'constant':
        net.apply(weights_init_constant)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler



def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[], noise=False, n_classes=3, y_x=1):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnetUp_9blocks':
        netG = ResnetGeneratorconv(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids, noise=noise)
    elif which_model_netG == 'resnetUp_6blocks':
        netG = ResnetGeneratorconv(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids, noise=noise, y_x=y_x)
    elif which_model_netG == 'resnetUp_9blocks_masked':
        netG = ResnetGeneratorMaskedconv(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids, n_classes=n_classes)
    elif which_model_netG == 'resnetUp_6blocks_masked':
        netG = ResnetGeneratorMaskedconv(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids, n_classes=n_classes)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' %which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[],y_x=1):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids,y_x=y_x)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD


def define_STN(input_nc, res=64, gpu_ids=[],y_x=1, STN_model=''):
    stn = None
    use_gpu = len(gpu_ids) > 0
    if res == 64:
        n_blocks=0
    elif res==128:
        n_blocks = 1
    elif res==256:
        n_blocks = 2
    else:
        raise NotImplementedError("STN not defined for this image resolution:%s"%res)
    if STN_model =='deep':
        stn = DeepSpatialTransformer(input_nc, n_blocks, gpu_ids,y_x)
    else:
        stn = SpatialTransformer(input_nc, n_blocks, gpu_ids,y_x)
    if use_gpu:
        assert(torch.cuda.is_available())
        stn.cuda(gpu_ids[0])
    # no weight initialization!!

    return stn


def define_AFN(model, input_nc=3, view_dim=19, flow_mask=0, res=128, init_type='normal', use_dropout=False, gpu_ids=[]):
    # Encoder-Decoder network for the "view synthesis by appreance flow" paper:
    netAFN = None
    if model=='fc':
        netAFN = DOAFNModel(view_dim=view_dim, flow_mask=flow_mask,res=res, disjoint_last_bottle='bottle', gpu_ids=gpu_ids)
    elif model=='fullyConv':
        netAFN = AFNconvModel(view_dim=view_dim, flow_mask=flow_mask,disjoint_last_bottle='bottle', gpu_ids=gpu_ids)
    elif model=='DOAFN':
        netAFN = DOAFNModel(view_dim=view_dim, flow_mask=flow_mask, res=res, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif model=='DOAFNCompose':
        netAFN = DOAFNComposeModel(input_nc, res=res, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError("model not recognized:%s"%model)

    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())
        netAFN.cuda(gpu_ids[0])
    init_weights(netAFN, init_type=init_type)


    return netAFN



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class fc_layer(nn.Module):
    def __init__(self, input_nc, output_nc, gpu_ids=[]):
        super(fc_layer, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.gpu_ids = gpu_ids
        self.model = [nn.Linear(input_nc, output_nc),
                      nn.ReLU(True)]


        # self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/

class ResnetGeneratorconv(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', noise=False, y_x=1):
        assert(n_blocks >= 0)
        super(ResnetGeneratorconv, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.noise = noise
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        #128 #256
        model1 = [nn.ReflectionPad2d(3), #134 #262
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias), #128 #256
                 norm_layer(ngf),
                 nn.ReLU(True)]

        mult = 1
        if y_x == 2:
            model1 += [nn.Conv2d(ngf *mult, ngf *mult, kernel_size=(1,3), 
                                    stride=(1,2), padding=(0,1), bias=use_bias),#128 #128
                          norm_layer(ngf * mult),
                          nn.ReLU(True)]

        model1 += [nn.Conv2d(ngf *mult, ngf *mult* 2, kernel_size=3, 
                                stride=2, padding=1, bias=use_bias),#64 #64
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]


        mult = 2
        if noise:
            nc_in = ngf * mult + int(ngf * mult/4.0)
        else:
            nc_in = ngf * mult 
        model2 = [nn.Conv2d(nc_in, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias), #32 #64
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        n_downsampling = 2
        mode = 'nearest'
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model2 += [nn.Upsample(scale_factor=2, mode=mode),
                        nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        nn.ReLU(True)]
        if y_x == 2:
            model2 += [nn.ConvTranspose2d(int(ngf * mult/2), int(ngf * mult / 2),
                                     kernel_size=(1,3), stride=(1,2),
                                     padding=(0,1), output_padding=(0,1),
                                     bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.ReLU(True)]

        model2 += [nn.ReflectionPad2d(3)]
        model2 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model2 += [nn.Tanh()]


        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)

    def forward(self, input, z=[]):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            x1 = nn.parallel.data_parallel(self.model1, input, self.gpu_ids)
            if self.noise:
                x1 = torch.cat([x1,z], 1)
            return nn.parallel.data_parallel(self.model2, x1, self.gpu_ids)

        else:
            x1 = self.model1(input)
            if self.noise:
                x1 = torch.cat([x1,z], 1)
            return self.model2(x1)


class ResnetGeneratorMaskedconv(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', n_classes=3):
        assert(n_blocks >= 0)
        super(ResnetGeneratorMaskedconv, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        #128 x 128 --> 134 x 134 --> 128 x 128
        encoder = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        mult = 1
        #128 x 128 --> 64 x 64
        encoder += [nn.Conv2d(ngf *mult, ngf *mult* 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2
        nc_in = ngf * mult 
        # 64  x 64 --> 32 x 32
        encoder += [nn.Conv2d(nc_in, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        n_downsampling = 2
        mult = 2**n_downsampling
        for i in range(n_blocks):
            encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        decoder= []
        mode = 'nearest'
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            # 32 x 32 --> 64 x 64
            # 64 x 64 --> 128 x 128
            decoder += [nn.Upsample(scale_factor=2, mode=mode),
                        nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        nn.ReLU(True)]

        # 128 x 128 --> 134 x 134 --> 128 x 128
        decoder += [nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                    nn.Tanh()]
        decoder_m= []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            decoder_m += [nn.Upsample(scale_factor=2, mode=mode),
                        nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        nn.ReLU(True)]            

        decoder_m += [nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, n_classes, kernel_size=7, padding=0)]


        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        self.decoder_m = nn.Sequential(*decoder_m)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            x1 = nn.parallel.data_parallel(self.encoder, input, self.gpu_ids)
            output = nn.parallel.data_parallel(self.decoder, x1, self.gpu_ids)
            mask = nn.parallel.data_parallel(self.decoder_m, x1, self.gpu_ids)
            return output, mask

        else:
            x1 = self.encoder(input)
            output = self.decoder(x1)
            mask = self.decoder_m(x1)
            return output,mask

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out




# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[],y_x=1):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        #128 #256
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),#63 #128
            nn.LeakyReLU(0.2, True)
        ]

        if y_x==2:
            sequence += [
                nn.Conv2d(ndf, ndf, kernel_size=(1,kw), stride=(1,2), padding=(0,padw)),#63 #63
                nn.LeakyReLU(0.2, True)
            ] 

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)



#Relative Spatial Transformer Network
class SpatialTransformer(nn.Module):
    def __init__(self, input_nc, n_blocks=0, gpu_ids=[],y_x=1):
        super(SpatialTransformer, self).__init__()
        self.gpu_ids = gpu_ids
        self.input_nc = input_nc
        self.n_blocks = n_blocks
        # Spatial transformer localization-network
        #128X128
        self.localization = []
        self.localization += [nn.Conv2d(input_nc, 8, kernel_size=7),#122 #(122,250)
            nn.MaxPool2d(2, stride=2), #62 , #(62,126)
            nn.ReLU(True)]
        if y_x == 2:
            self.localization += [nn.Conv2d(8, 8, kernel_size=(1,5)),#(62, 122)
                nn.MaxPool2d((1,2), stride=(1,2)), #(62, 62)
                nn.ReLU(True)]
        self.localization += [nn.Conv2d(8, 10, kernel_size=5), #58 #(58, 58)
            nn.MaxPool2d(2, stride=2), #30
            nn.ReLU(True),
            nn.Conv2d(10, 30, kernel_size=5), #26
            nn.MaxPool2d(2, stride=2), #14
            nn.ReLU(True)]

        self.out_dim = 4
        for i in range(0,n_blocks):
            self.localization += [nn.Conv2d(30, 30, kernel_size=5),#10
                                nn.MaxPool2d(2, stride=2),nn.ReLU(True)] #4

        self.localization = nn.Sequential(*self.localization)

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(30 * self.out_dim * self.out_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 4)
        )


        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0])

    def forward(self, input, no_translatoin=False):
        h = input.size(2)
        w = input.size(3)
        if len(self.gpu_ids)  and isinstance(input.data, torch.cuda.FloatTensor):
            mask = torch.Tensor(input.size()).fill_(1).cuda(self.gpu_ids[0], async=True)
        ONES = mask.clone()
        mask.index_fill_(2, LongTensor([0, h-1]).cuda(self.gpu_ids[0]), 0)
        mask.index_fill_(3, LongTensor([0, w-1]).cuda(self.gpu_ids[0]), 0)
        mask = Variable(mask)

        # misc.imsave('/home/sazadi/projects/objectComposition-Pytorch/mask.png', mask.data.cpu().numpy().transpose(1,2,0))
        input = torch.mul(input,mask) + (Variable(ONES) - mask)

        if len(self.gpu_ids)  and isinstance(input.data, torch.cuda.FloatTensor):
            xs = nn.parallel.data_parallel(self.localization, input, self.gpu_ids)
        else:
            xs = self.localization(input)

        xs = xs.view(-1, 30 * self.out_dim * self.out_dim)
        ind1 = Variable(LongTensor(range(0,2)))
        ind2 = Variable(LongTensor(range(2,4)))
        inp1 = Variable(LongTensor(range(0,int(input.size(1)/2))))
        inp2 = Variable(LongTensor(range(int(input.size(1)/2),input.size(1))))

        if len(self.gpu_ids)  and isinstance(input.data, torch.cuda.FloatTensor):        
            theta = nn.parallel.data_parallel(self.fc_loc, xs, self.gpu_ids)
            ind1 = ind1.cuda()
            ind2 = ind2.cuda()
            inp1 = inp1.cuda()
            inp2 = inp2.cuda()

        else:
            theta = self.fc_loc(xs)
        theta = theta.view(-1, 4, 3)


        theta_1 = index_select(theta,1, ind1)
        theta_2 = index_select(theta,1, ind2)
        if no_translatoin:
            translation_mat = Variable(torch.ones((2,3)).cuda(),requires_grad=False)
            translation_mat[:,2] = 0
            theta_1_linear = torch.mul(theta_1, translation_mat)
            theta_2_linear = torch.mul(theta_2, translation_mat)


        input_1 = index_select(input, 1, inp1)
        input_2 = index_select(input, 1, inp2)
        grid_1 = F.affine_grid(theta_1, input_1.size())
        grid_2 = F.affine_grid(theta_2, input_2.size())

        x1 = F.grid_sample(input_1, grid_1, padding_mode="border")
        x2 = F.grid_sample(input_2, grid_2, padding_mode="border")
        if no_translatoin:
            grid_1 = F.affine_grid(theta_1_linear, input_1.size())
            grid_2 = F.affine_grid(theta_2_linear, input_2.size())
            x1_linear = F.grid_sample(input_1, grid_1, padding_mode="border")
            x2_linear = F.grid_sample(input_2, grid_2, padding_mode="border")
            return x1,x2, x1_linear, x2_linear
        else:
            return x1,x2


# Deeper Relative Spatial Transformer Network
class DeepSpatialTransformer(nn.Module):
    def __init__(self, input_nc, n_blocks=0, gpu_ids=[],y_x=1):
        super(DeepSpatialTransformer, self).__init__()
        self.gpu_ids = gpu_ids
        self.input_nc = input_nc
        self.n_blocks = n_blocks
        # Spatial transformer localization-network
        #128X128
        self.localization = []
        self.localization += [nn.Conv2d(input_nc, 32, kernel_size=7),#122 #(122,250)
            nn.MaxPool2d(2, stride=2), #62 , #(62,126)
            nn.ReLU(True)]
        if y_x == 2:
            self.localization += [nn.Conv2d(32, 32, kernel_size=(1,5)),#(62, 122)
                nn.MaxPool2d((1,2), stride=(1,2)), #(62, 62)
                nn.ReLU(True)]
        self.localization += [nn.Conv2d(32, 64, kernel_size=5), #58 #(58, 58)
            nn.MaxPool2d(2, stride=2), #30
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=5), #26
            nn.MaxPool2d(2, stride=2), #14
            nn.ReLU(True)]

        self.out_dim = 4
        for i in range(0,n_blocks):
            self.localization += [nn.Conv2d(128, 128, kernel_size=5),#10
                                nn.MaxPool2d(2, stride=2),nn.ReLU(True)] #4

        self.localization = nn.Sequential(*self.localization)

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * self.out_dim * self.out_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 3*4)
        )


        self.fc_loc[4].weight.data.fill_(0)
        self.fc_loc[4].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0])

    def forward(self, input, no_translatoin=False):
        h = input.size(2)
        w = input.size(3)
        if len(self.gpu_ids)  and isinstance(input.data, torch.cuda.FloatTensor):
            mask = torch.Tensor(input.size()).fill_(1).cuda(self.gpu_ids[0], async=True)
        ONES = mask.clone()
        mask.index_fill_(2, LongTensor([0, h-1]).cuda(self.gpu_ids[0]), 0)
        mask.index_fill_(3, LongTensor([0, w-1]).cuda(self.gpu_ids[0]), 0)
        mask = Variable(mask)

        # misc.imsave('/home/sazadi/projects/objectComposition-Pytorch/mask.png', mask.data.cpu().numpy().transpose(1,2,0))
        input = torch.mul(input,mask) + (Variable(ONES) - mask)

        if len(self.gpu_ids)  and isinstance(input.data, torch.cuda.FloatTensor):
            xs = nn.parallel.data_parallel(self.localization, input, self.gpu_ids)
        else:
            xs = self.localization(input)

        xs = xs.view(-1, 128 * self.out_dim * self.out_dim)
        ind1 = Variable(LongTensor(range(0,2)))
        ind2 = Variable(LongTensor(range(2,4)))
        inp1 = Variable(LongTensor(range(0,int(input.size(1)/2))))
        inp2 = Variable(LongTensor(range(int(input.size(1)/2),input.size(1))))

        if len(self.gpu_ids)  and isinstance(input.data, torch.cuda.FloatTensor):        
            theta = nn.parallel.data_parallel(self.fc_loc, xs, self.gpu_ids)
            ind1 = ind1.cuda()
            ind2 = ind2.cuda()
            inp1 = inp1.cuda()
            inp2 = inp2.cuda()

        else:
            theta = self.fc_loc(xs)
        theta = theta.view(-1, 4, 3)


        theta_1 = index_select(theta,1, ind1)
        theta_2 = index_select(theta,1, ind2)
        if no_translatoin:
            translation_mat = Variable(torch.ones((2,3)).cuda(),requires_grad=False)
            translation_mat[:,2] = 0
            theta_1_linear = torch.mul(theta_1, translation_mat)
            theta_2_linear = torch.mul(theta_2, translation_mat)


        input_1 = index_select(input, 1, inp1)
        input_2 = index_select(input, 1, inp2)
        grid_1 = F.affine_grid(theta_1, input_1.size())
        grid_2 = F.affine_grid(theta_2, input_2.size())

        x1 = F.grid_sample(input_1, grid_1, padding_mode="border")
        x2 = F.grid_sample(input_2, grid_2, padding_mode="border")
        if no_translatoin:
            grid_1 = F.affine_grid(theta_1_linear, input_1.size())
            grid_2 = F.affine_grid(theta_2_linear, input_2.size())
            x1_linear = F.grid_sample(input_1, grid_1, padding_mode="border")
            x2_linear = F.grid_sample(input_2, grid_2, padding_mode="border")
            return x1,x2, x1_linear, x2_linear
        return x1,x2


#modified reimplementation of the view synthesis by appearance flow paper: encoder network
class AFNconvModel(nn.Module):
    def __init__(self, view_dim=19, flow_mask=0, norm_layer=nn.BatchNorm2d, padding_type='zero', use_dropout=False, use_bias=False, n_blocks=3,disjoint_last_bottle='bottle', gpu_ids=[]):
        super(AFNconvModel, self).__init__()
        self.gpu_ids = gpu_ids
        #flow or mask prediction network
        self.flow_mask = flow_mask
        self.disjoint_last_bottle = disjoint_last_bottle

        #Encoder for input view image
        sequence_conv_inp = [
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            norm_layer(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            norm_layer(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            norm_layer(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm_layer(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm_layer(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            norm_layer(512),
            nn.ReLU(True)
        ]

        self.encoder_conv_inp = nn.Sequential(*sequence_conv_inp)

        sequence_up = []
        #add resent blocks for the bottleneck part
        for i in range(n_blocks):
            sequence_up += [ResnetBlock(512+view_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
        #Decoder deconvolutions/upsampling:
        sequence_up += [
            nn.ConvTranspose2d(512+view_dim, 256, kernel_size=3, stride=3, padding=2, output_padding=1),
            norm_layer(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            norm_layer(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            norm_layer(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            norm_layer(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            norm_layer(16),
            nn.ReLU(True)
        ]

        if disjoint_last_bottle=='bottle':
            sequence_up_mask = []
            #add resent blocks for the bottleneck part
            for i in range(n_blocks):
                sequence_up_mask += [ResnetBlock(512+view_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            
            #Decoder deconvolutions/upsampling:
            sequence_up_mask += [
                nn.ConvTranspose2d(512+view_dim, 256, kernel_size=3, stride=3, padding=2, output_padding=1),
                norm_layer(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
                norm_layer(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
                norm_layer(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
                norm_layer(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
                norm_layer(16),
                nn.ReLU(True)
            ]
            self.decoder_up_mask = nn.Sequential(*sequence_up_mask)

        flow_out = [nn.ConvTranspose2d(16, 2, kernel_size=2, stride=1, padding=1),
                    nn.Tanh()]
        mask_out = [nn.ConvTranspose2d(16, 1, kernel_size=2, stride=1, padding=1),
                    nn.Sigmoid()]
        self.flow_out = nn.Sequential(*flow_out)
        self.mask_out = nn.Sequential(*mask_out)

        self.decoder_up = nn.Sequential(*sequence_up)

    def forward(self, img, view):
        # view = view.view([view.size(0), view.size(1)*view.size(2)]).type(torch.FloatTensor)
        if len(self.gpu_ids) and isinstance(img.data, torch.cuda.FloatTensor):
            img = nn.parallel.data_parallel(self.encoder_conv_inp, img, self.gpu_ids)
            view = view.unsqueeze(2).repeat(1, 1, 4, 4).type(torch.cuda.FloatTensor)
            # normalize img features
            # img = F.normalize(img, p=2, dim=1)
            input = torch.cat((img, view), dim=1)
            pred = nn.parallel.data_parallel(self.decoder_up, input, self.gpu_ids)
            flow = nn.parallel.data_parallel(self.flow_out, pred, self.gpu_ids)
            if self.disjoint_last_bottle == 'bottle':
                pred = nn.parallel.data_parallel(self.decoder_up_mask, input, self.gpu_ids)
            mask = nn.parallel.data_parallel(self.mask_out, pred, self.gpu_ids)

        else:
            img = self.encoder_inp(img)
            # img = F.normalize(img, p=2, dim=1)
            view = view.repeat(view.size(0), view.size(1), 4, 4)
            input = torch.cat((img, view), dim=1)
            pred = self.decoder_up(input)
            flow = self.flow_out(pred)
            if self.disjoint_last_bottle=='bottle':
                pred = self.decoder_up_mask(input)
            mask = self.mask_out(pred)

        return flow,mask


# reimplementation of the dofan model in torch
#https://github.com/silverbottlep/tvsn/blob/49820d819a3d3588e7c4ff3c3c9c4698a5593c53/tvsn/code/models/DOAFN_SYM_256.lua
class DOAFNModel(nn.Module):
    def __init__(self, view_dim=19, flow_mask=0, res=128, norm_layer=nn.BatchNorm2d, padding_type='zero', use_dropout=False, use_bias=False, n_blocks=3, disjoint_last_bottle='last', gpu_ids=[]):
        super(DOAFNModel, self).__init__()
        self.gpu_ids = gpu_ids
        #flow or mask prediction network
        self.flow_mask = flow_mask
        self.disjoint_last_bottle =disjoint_last_bottle

        #Encoder for input view image
        if res==256:
            s = 2
        elif res==128:
            s = 1
        sequence_conv_inp = [
            # 128 x 128 x 3 --> 64 x 64 x 16 
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            norm_layer(16),
            nn.ReLU(True),
            # 64 x 64 x 16 --> 32 x 32 x 32
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            norm_layer(32),
            nn.ReLU(True),
            # 32 x 32 x 32 --> 16 x 16 x 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            norm_layer(64),
            nn.ReLU(True),
            # 16 x 16 x 64 --> 8 x 8 x 128          
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm_layer(128),
            nn.ReLU(True),
            # 8 x 8 x 64 --> 4 x 4 x 256 
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm_layer(256),
            nn.ReLU(True),
            # 4 x 4 x 256 --> 4 x 4 x 512 
            nn.Conv2d(256, 512, kernel_size=3, stride=s, padding=1),
            norm_layer(512),
            nn.ReLU(True)
        ]

        self.encoder_conv_inp = nn.Sequential(*sequence_conv_inp)

        sequence_fc_inp = [
            nn.Linear(512*4*4, 2048),
            nn.ReLU(True),
        ]

        self.encoder_fc_inp = nn.Sequential(*sequence_fc_inp)

        #Encoder for viewpoint transformation
        sequence_view = [
            nn.Linear(view_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
        ]
        self.encoder_view = nn.Sequential(*sequence_view)

        #Decoder fc layers 
        #flow field/mask prediction
        if use_dropout:
            p = 0.5
        else:
            p = 0
        sequence_fc = [
            nn.Linear(2048+256, 2048),
            nn.ReLU(True),
            nn.Dropout(p),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(p),
            nn.Linear(2048, 512*4*4),
            nn.ReLU(True),
            ]
        self.decoder_fc = nn.Sequential(*sequence_fc)

        #Decoder deconvolutions/upsampling:
        mode = 'nearest'
        sequence_up = [
            # 4 x 4 x 512 --> 8 x 8 x 256
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            norm_layer(256),
            nn.ReLU(True),
            # 8 x 8 x 256 --> 16 x 16 x 128
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            norm_layer(128),
            nn.ReLU(True),
            # 16 x 16 x 128 --> 32 x 32 x 64
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            norm_layer(64),
            nn.ReLU(True),
            # 32 x 32 x 64 --> 64 x 64 x 32
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            norm_layer(32),
            nn.ReLU(True),
            # 64 x 64 x 32 --> 128 x 128 x 16
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
            norm_layer(16),
            nn.ReLU(True)
            ]
        self.decoder_up = nn.Sequential(*sequence_up)

        if disjoint_last_bottle=='bottle':
            sequence_up_mask = [
                # 4 x 4 x 512 --> 8 x 8 x 256
                nn.Upsample(scale_factor=2, mode=mode),
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                norm_layer(256),
                nn.ReLU(True),
                # 8 x 8 x 256 --> 16 x 16 x 128
                nn.Upsample(scale_factor=2, mode=mode),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                norm_layer(128),
                nn.ReLU(True),
                # 16 x 16 x 128 --> 32 x 32 x 64
                nn.Upsample(scale_factor=2, mode=mode),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                norm_layer(64),
                nn.ReLU(True),
                # 32 x 32 x 64 --> 64 x 64 x 32
                nn.Upsample(scale_factor=2, mode=mode),
                nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
                norm_layer(32),
                nn.ReLU(True),
                # 64 x 64 x 32 --> 128 x 128 x 16
                nn.Upsample(scale_factor=2, mode=mode),
                nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
                norm_layer(16),
                nn.ReLU(True)
            ] 
            self.decoder_up_mask = nn.Sequential(*sequence_up_mask)

        if res==256:
            flow_out = [nn.Upsample(scale_factor=2, mode=mode)]
            mask_out = [nn.Upsample(scale_factor=2, mode=mode)]
        else:
            flow_out = []
            mask_out = []
        flow_out += [
            # 128 x 128 x 16 --> 128 x 128 x 2
            nn.Conv2d(16, 2, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
            ]

        mask_out += [
            # 128 x 128 x 16 --> 128 x 128 x 2
            nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        ]

        self.flow_out = nn.Sequential(*flow_out)
        self.mask_out = nn.Sequential(*mask_out)

    def forward(self, img, view):
        view = view.view([view.size(0), view.size(1)*view.size(2)]).type(torch.FloatTensor)
        if len(self.gpu_ids) and isinstance(img.data, torch.cuda.FloatTensor):
            view = view.cuda(self.gpu_ids[0], async=True)
            img = nn.parallel.data_parallel(self.encoder_conv_inp, img, self.gpu_ids)
            img = img.view([img.size(0), 512*4*4])
            img = nn.parallel.data_parallel(self.encoder_fc_inp, img, self.gpu_ids)

            view = nn.parallel.data_parallel(self.encoder_view, view, self.gpu_ids)
        else:
            img = self.encoder_inp(img)
            view = self.encoder_view(view)

        
        encoder_output = torch.cat([img, view],1)
        if len(self.gpu_ids) and isinstance(img.data, torch.cuda.FloatTensor):
            decoder_inp = nn.parallel.data_parallel(self.decoder_fc, encoder_output, self.gpu_ids)
        else:
            decoder_inp = self.decoder_fc(encoder_output)

        decoder_inp = decoder_inp.view([decoder_inp.size(0), 512, 4, 4])

        if len(self.gpu_ids) and isinstance(img.data, torch.cuda.FloatTensor):
            pred = nn.parallel.data_parallel(self.decoder_up, decoder_inp, self.gpu_ids)
            flow = nn.parallel.data_parallel(self.flow_out, pred, self.gpu_ids)
            if self.disjoint_last_bottle=='bottle':
                pred = nn.parallel.data_parallel(self.decoder_up_mask, decoder_inp, self.gpu_ids)
            mask = nn.parallel.data_parallel(self.mask_out, pred, self.gpu_ids)
        else:
            pred = self.decoder_up(decoder_inp)
            flow = self.flow_out(pred)
            if disjoint_last_bottle=='bottle':
                pred = self.decoder_up_mask(decoder_inp)
            mask = self.mask_out(pred)
        return flow, mask




# reimplementation of the dofan model in torch
#https://github.com/silverbottlep/tvsn/blob/49820d819a3d3588e7c4ff3c3c9c4698a5593c53/tvsn/code/models/DOAFN_SYM_256.lua
class DOAFNComposeModel(nn.Module):
    def __init__(self, input_nc, res=128, norm_layer=nn.BatchNorm2d, padding_type='zero', use_dropout=False, use_bias=False, n_blocks=3, gpu_ids=[]):
        super(DOAFNComposeModel, self).__init__()
        self.gpu_ids = gpu_ids
        self.input_nc = input_nc

        #Encoder for input view image
        if res==256:
            s = 2
        elif res==128:
            s = 1
        sequence_enc = [
            # 128 x 128 x 3 --> 64 x 64 x 16 
            nn.Conv2d(input_nc, 16, kernel_size=5, stride=2, padding=2),
            norm_layer(16),
            nn.ReLU(True),
            # 64 x 64 x 16 --> 32 x 32 x 32
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            norm_layer(32),
            nn.ReLU(True),
            # 32 x 32 x 32 --> 16 x 16 x 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            norm_layer(64),
            nn.ReLU(True),
            # 16 x 16 x 64 --> 8 x 8 x 128          
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm_layer(128),
            nn.ReLU(True),
            # 8 x 8 x 64 --> 4 x 4 x 256 
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm_layer(256),
            nn.ReLU(True),
            # 4 x 4 x 256 --> 4 x 4 x 512 
            nn.Conv2d(256, 512, kernel_size=3, stride=s, padding=1),
            norm_layer(512),
            nn.ReLU(True)
        ]
        self.encoder = nn.Sequential(*sequence_enc)
        #add resent blocks for the bottleneck part
        sequence_dec_f = []
        for i in range(n_blocks):
            sequence_dec_f += [ResnetBlock(512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
        sequence_dec_m = []
        for i in range(n_blocks):
            sequence_dec_m += [ResnetBlock(512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        #Decoder deconvolutions/upsampling:
        mode = 'nearest'
        sequence_dec = [
            # 4 x 4 x 512 --> 8 x 8 x 256
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            norm_layer(256),
            nn.ReLU(True),
            # 8 x 8 x 256 --> 16 x 16 x 128
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            norm_layer(128),
            nn.ReLU(True),
            # 16 x 16 x 128 --> 32 x 32 x 64
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            norm_layer(64),
            nn.ReLU(True),
            # 32 x 32 x 64 --> 64 x 64 x 32
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            norm_layer(32),
            nn.ReLU(True),
            # 64 x 64 x 32 --> 128 x 128 x 16
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
            norm_layer(16),
            nn.ReLU(True)
            ]

        sequence_dec_f += sequence_dec
        sequence_dec_m += sequence_dec

        if res==256:
            sequence_dec_f += [nn.Upsample(scale_factor=2, mode=mode)]
            sequence_dec_m += [nn.Upsample(scale_factor=2, mode=mode)]
        sequence_dec_f += [
            # 128 x 128 x 16 --> 128 x 128 x 2
            nn.Conv2d(16, 2, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
            ]

        sequence_dec_m += [
            # 128 x 128 x 16 --> 128 x 128 x 2
            nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        ]

        self.decoder_m = nn.Sequential(*sequence_dec_m)
        self.decoder_f = nn.Sequential(*sequence_dec_f)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            emb = nn.parallel.data_parallel(self.encoder, input, self.gpu_ids)
            flow = nn.parallel.data_parallel(self.decoder_f, emb, self.gpu_ids)

            mask = nn.parallel.data_parallel(self.decoder_m, emb, self.gpu_ids)

        else:
            emb = elf.encoder(input)
            flow = self.decoder_f(emb)
            mask = self.decoder_m(emb)

        return flow, mask

