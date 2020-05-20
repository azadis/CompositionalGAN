# ========================================================
# Compositional GAN
# Options for the base parameters
# By Samaneh Azadi
# ========================================================


import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--datalist', required=True, help='path to a txt file with each line: path_A, path_B, azimuth')
        self.parser.add_argument('--datalist_test', required=True, help='path to a txt file with each line: path_A, path_B, azimuth')
        self.parser.add_argument('--test_path_azi', type=str, default='dataset/chair_table/test_random_azi.pkl', help='where is the pickle file including fixed random azimuth angles for test?')
        self.parser.add_argument('--num_az', type=int, default=36, help='number of azimuth angles in the viewpoint transformation')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSizeX', type=int, default=128, help='scale images to this size')
        self.parser.add_argument('--fineSizeX', type=int, default=128, help='then crop to this size')
        self.parser.add_argument('--loadSizeY', type=int, default=128, help='scale images to this size')
        self.parser.add_argument('--fineSizeY', type=int, default=128, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--G1_completion', type=int, default=1, help='completion on obj1')
        self.parser.add_argument('--G2_completion', type=int, default=1, help='completion on obj2')
        self.parser.add_argument('--Thresh1', type=float, default=0.9, help='masking threshold for obj1')
        self.parser.add_argument('--Thresh2', type=float, default=0.9, help='masking threshold for obj2')
        self.parser.add_argument('--which_epoch_completion', type=str, default='0', help='which epoch to load G_completion model? set to latest to use latest cached model')
        self.parser.add_argument('--which_epoch_STN', type=str, default='0', help='which epoch to load STN_c model? set to latest to use latest cached model')
        self.parser.add_argument('--which_epoch_AFN', type=str, default='100', help='which epoch to load AFN model? set to latest to use latest cached model')
        self.parser.add_argument('--which_model_netD', type=str, default='n_layers', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resnetUp_6blocks', help='selects model to use for netG')
        self.parser.add_argument('--which_model_AFN', type=str, default='DOAFNCompose', help='which model to use for AFN model? fc or fullyconv?')
        self.parser.add_argument('--STN_model', type=str, default='', help='which model to use for STN model? deep or vae or ?')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [comp_decomp | comp_decomp_aligned | compose | decompose]')
        self.parser.add_argument('--model', type=str, default='objCompose',
                                 help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=20000, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--noise', action='store_true', help='if specified, add noise to generator')
        self.parser.add_argument('--conditional', action='store_true', help='if specified, conditional GAN discriminators')
        self.parser.add_argument('--random_view', action='store_true', help='if specified, randomly view point of obj1 wrt obj2')
        self.parser.add_argument('--decomp', action='store_true', help='if specified, do decomposition too.')
        self.parser.add_argument('--img_completion', action='store_true', help='if specified, do image completion')
        self.parser.add_argument('--erosion', action='store_true', help='if specified, erode the borders of the real masks (useful for test time')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
