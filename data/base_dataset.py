import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch import index_select, LongTensor, nn
from scipy import misc
from scipy.misc import imread
import numpy as np

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt, pad=False, size=(128,128)):
    transform_list = []
    if pad:
        Im_r = size[0]
        Im_c = size[1]
        osize = max(Im_r, Im_c)
        padding = int(abs(Im_c-Im_r)/2)
        if (Im_r<Im_c): 
            padding = (padding,0,padding,0)
        else:
            padding = (0,padding,0,padding)

        transform_list.append(transforms.Pad(padding, fill=(255,255,255)))
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    return transforms.Compose(transform_list)

def transform2NormalTens():
    transform_list = []
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)

def BorderOne(img):
    print(img.size())
    h = img.size(1)
    w = img.size(2)
    mask = torch.Tensor(img.size()).fill_(1)
    mask.index_fill_(1, LongTensor([0, h-1]), 0)
    mask.index_fill_(2, LongTensor([0, w-1]), 0)
    img = torch.mul(img,mask) + (torch.Tensor(img.size()).fill_(1) - mask)
    return img

def ChangeDirection(path):
    im0 = imread(path)
    im = np.mean(im0,2)
    mask = im<240
    h,w = mask.shape
    ul = np.mean(mask[0:h//2, 0:w//2])
    ur = np.mean(mask[0:h//2, w//2:])
    ll = np.mean(mask[h//2:, 0:w//2])
    lr = np.mean(mask[h//2:, w//2:])
    diag1 = ul + lr
    diag2 = ur + ll
    if (diag1 > diag2):
        im0 = im0[:,::-1,:].copy()
    im = Image.fromarray(im0)
    return im




