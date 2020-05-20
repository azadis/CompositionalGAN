import torch
from torch.autograd import Variable
from torch import index_select, LongTensor, nn
import numpy as np


def Gaussian2D(dim, mu_x, mu_y, sigma_x, sigma_y,gpu_ids=[]):
    batch = mu_x.size(0)
    x = torch.Tensor(range(dim)).cuda(gpu_ids[0], async=True)
    x = x.repeat(dim, 1)
    y = x.transpose(1,0)
    x = x.unsqueeze(0).repeat(batch, 1, 1)
    y = y.unsqueeze(0).repeat(batch,1, 1).cuda(gpu_ids[0], async=True)
    x = Variable(x.cuda())
    y = Variable(y.cuda())
    mu_x = mu_x.unsqueeze(2).repeat(1, dim, dim)
    mu_y = mu_y.unsqueeze(2).repeat(1, dim, dim)
    sigma_x = sigma_x.unsqueeze(2).repeat(1, dim, dim)
    sigma_y = sigma_y.unsqueeze(2).repeat(1, dim, dim)
    z = (-1/2*(((x-mu_x)**2)*(1/sigma_x)+((y-mu_y)**2)*(1/sigma_y)))
    return z

def slidingWindow(im, stride, gpu_ids=[]):
    dim = im.size(2)
    batch = im.size(0)
    im_ = im.data
    scales = [1/6, 1/4, 1/3, 1/2]
    ratios = [1/2, 2/1, 1/3, 3/1, 1/1, 2/3, 3/2]

    all_s = 0
    for scale in scales:
        for ratio in ratios:
                per_stride = len(range(0, int(dim), int(stride)))
                all_s += (per_stride)**2

    cropped_im = (torch.Tensor(batch*all_s,\
                            im.size(1), im.size(2),im.size(3)).cuda(gpu_ids[0], async=True).fill_(1))

    box_info = (torch.Tensor(all_s,im.size(2),im.size(3)).cuda(gpu_ids[0], async=True).fill_(0))

    sr = 0
    i = 0
    for scale in scales:
        for ratio in ratios:
            area = scale * (dim**2)
            W = int((area/ratio)**0.5)
            H = int(W * ratio)
            x_starts = range(0, int(dim), int(stride))
            y_starts = range(0, int(dim), int(stride))
            for x_s in x_starts:
                for y_s in y_starts:
                    im_x = index_select(im_,2,(LongTensor(range(x_s, min(x_s+H, im.size(2)))).cuda(gpu_ids[0], async=True)))
                    im_x_y = index_select(im_x,3,(LongTensor(range(y_s, min(y_s+W, im.size(3)))).cuda(gpu_ids[0], async=True)))
                    cropped_im[(batch*i):batch*(i+1), :,x_s:min(x_s+H, im.size(2)) ,y_s:min(y_s+W, im.size(3))] = im_x_y
                    box_info[i,x_s:min(x_s+H, im.size(2)) ,y_s:min(y_s+W, im.size(3))] = 1
                    i += 1
            sr += 1

            
    batch_index = np.tile(range(0, batch*len(x_starts)*len(y_starts)*len(scales)*len(ratios), batch), batch) +\
                  np.repeat(range(0, batch), len(x_starts)*len(y_starts)*len(scales)*len(ratios)) 

    cropped_im = (index_select(cropped_im, 0, (LongTensor(batch_index).cuda(gpu_ids[0], async=True))))
    box_info = torch.sum(box_info,0)
 


    return cropped_im, box_info








