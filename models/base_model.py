import os
import torch
import util.util as util

# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.model_names = []
        self.epoch_labels = []
        self.optimizers = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    def save_networks(self, model_names, epoch_labels):
        assert(len(model_names) == len(epoch_labels))
        for i,name in enumerate(model_names):
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                save_filename = '%s_net_%s.pth' % (epoch_labels[i], name)
                save_path = os.path.join(self.save_dir, save_filename)
                if len(self.gpu_ids) and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)


    # # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))


    # helper loading function that can be used by subclasses
    def load_networks(self):
        assert(len(self.model_names) == len(self.epoch_labels))
        for i,name in enumerate(self.model_names):
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                save_filename = '%s_net_%s.pth' % (self.epoch_labels[i], name)
                save_path = os.path.join(self.save_dir, save_filename)
                net.load_state_dict(torch.load(save_path))

    # define the optimizers
    def set_optimizers(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                setattr(self, 'optimizer_' + name, torch.optim.Adam(net.parameters(),
                                                lr=self.opt.lr, betas=(self.opt.beta1, 0.999)))
                self.optimizers.append(getattr(self, 'optimizer_' + name))


    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    # print network information
    def print_networks(self, verbose=False):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def tocuda(self, var_names):
        for name in var_names:
            if isinstance(name, str):
                var = getattr(self, name)
                setattr(self, name, var.cuda(self.gpu_ids[0], async=True))


    def tnsrs2ims(self, tensor_names):
        ims = []
        for name in tensor_names:
            if isinstance(name, str):
                var = getattr(self, name)
                ims.append(util.tensor2im(var.data))
        return ims




