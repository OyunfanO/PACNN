import os

from data import common

import numpy as np
import scipy.misc as misc
import scipy.io as sio
from scipy.misc import imresize

import torch
import torch.utils.data as data
import h5py
import cv2
import random

class SRData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0
        self.is_grey = args.is_grey

        if train:
            #mat = h5py.File('../PACNN/imdb_gray.mat')
            mat = h5py.File('../SIDD_GT/image_denoising.mat')
            mat_lr = h5py.File('../SIDD_N/image_denoising.mat')
            self.args.ext = 'mat'
            self.hr_data = mat['images']['labels'][:,:,:,:]
            self.lr_data = mat_lr['images']['labels'][:,:,:,:]
            self.num = self.hr_data.shape[0]
            #print(self.hr_data.shape)

        if self.split == 'test':
            self._set_filesystem(args.dir_data)

        self.images_hr = self._scan()
        #print(self.images_hr)



    def _scan(self):
        raise NotImplementedError
        '''
        if self.train:
            list_hr = [i for i in range(self.num)]
        else:
            list_hr = []
            # list_hr = []
            # # list_lr = [[] for _ in self.scale]
            for entry in os.scandir(self.dir_hr):
                filename = os.path.splitext(entry.name)[0]
                list_hr.append(os.path.join(self.dir_hr, filename + self.img_ext))
            list_hr.sort()
        return list_hr#[i for i in range(self.num)]#, list_lr
        '''
    #
    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    # def _name_hrbin(self):
    #     raise NotImplementedError

    # def _name_lrbin(self, scale):
    #     raise NotImplementedError

    def __getitem__(self, idx):
        hr, lr, filename = self._load_file(idx)
        #hr = common.set_channel(hr,self.args.n_colors,self.is_grey)
        #lr = common.set_channel(lr,self.args.n_colors,self.is_grey)
        if self.train:
            lr, hr, scale, nl = self._get_patch(hr, lr, filename)
            lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
            return lr_tensor, hr_tensor, filename, nl
        else:
            #scale = 2
            # scale = self.scale[self.idx_scale]
            lr, hr, scale, nl = self._get_patch(hr, lr, filename)
            lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
            return lr_tensor, hr_tensor, filename, nl


    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        # lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]

        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            hr = misc.imread(hr)
            lr = filename.replace("gt","noise")
            lr = misc.imread(lr)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            # lr = np.load(lr)
            hr = np.load(hr)
        elif self.args.ext == 'mat' or self.train:
            hr = self.hr_data[idx, :, :, :]
            hr = np.squeeze(hr.transpose((1, 2, 0)))
            lr = self.lr_data[idx, :, :, :]
            lr = np.squeeze(lr.transpose((1, 2, 0)))
            filename = str(idx) + '.png'
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return hr, lr, filename

    def _get_patch(self, hr, lr, filename):
        patch_size = self.args.patch_size

        if self.train:
            scale = self.scale[0]
            hr, lr = self.augment2(hr, lr)
            return lr, hr, scale, scale
        else:
            scale = self.scale[0]
            return lr, hr, scale, scale
            # lr = common.add_noise(lr, self.args.noise)

    def _get_patch_test(self, hr, scale):

        hr = np.squeeze(hr)
        ih, iw = hr.shape[0:2]

        #hr = hr.astype('uint8')
        #lr = imresize(imresize(hr, [int(ih/scale), int(iw/scale)], 'bicubic'), [ih, iw], 'bicubic')
        lr = cv2.resize(hr,(int(iw/scale),int(ih/scale)),interpolation=cv2.INTER_CUBIC)
        lr = cv2.resize(lr,(iw,ih),interpolation=cv2.INTER_CUBIC)
        lr = lr.clip(0,255)
        lr = np.expand_dims(lr, axis =2)
        hr = np.expand_dims(hr, axis =2)
        ih = ih // 8 * 8
        iw = iw // 8 * 8
        hr = hr[0:ih, 0:iw, :]
        lr = lr[0:ih, 0:iw, :]

        return lr, hr

    def augment(self, img, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img
        
    def augment2(self, img1, img2):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5
        if hflip: 
            img1 = img1[:, ::-1, :]
            img2 = img2[:, ::-1, :]
        if vflip: 
            img1 = img1[::-1, :, :]
            img2 = img2[::-1, :, :]
        if rot90: 
            img1 = img1.transpose(1, 0, 2)
            img2 = img2.transpose(1, 0, 2)
        return img1, img2


    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

