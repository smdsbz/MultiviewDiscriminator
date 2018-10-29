# -*- coding: utf-8 -*-

import torch
from torch.utils.data.dataset import Dataset
import skimage.transform as T

import numpy as np

import os

import matplotlib.image as mpimage

import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DigitDataset(Dataset):
    ''' Digits Dataset Class '''

    def __init__(self, path, grayscale=True, device=device):
        '''
        Args:
            path (string): path to all the `.png` images
        '''
        self.rootdir = path
        self.grayscale = grayscale
        self.device = device
        self.avail_digits = [ d for d in range(10) ]
        self.legal_rots = {
            'x': [ an for an in range(0, 360, 10) ],
            'y': [ an for an in range(90, -90, -10) ]
        }
        # get __len__() output
        self.len = len(self.avail_digits) * len(self.legal_rots['x']) * len(self.legal_rots['y'])

    def imgread(self, digit, xrot, yrot, grayscale=True):
        '''
        Return:
            (np.ndarray of (733, 1200[, 3]))
            If not grayscale, return pixel value matrix for RGB layers, else
            return grayscaled value. Pixel value in range [0.0, 1.0] inclusive.
        '''
        try:
            img = mpimage.imread(os.path.join(
                self.rootdir,
                '字母数字符号大全_pic{}_{}_{}.png' .format(digit, xrot, yrot)
            ))
        except FileNotFoundError:
            return None
        img = img[:, :, :3]     # discard alpha channel
        if grayscale:
            img = np.sum(img, axis=2) / 3.0
        return img

    def __getitem__(self, *args):
        '''
        Args:
            *args (tuple): if `len(args) == 3`, return the image with name
                '字母数字符号大全_pic{args[0]}_{args[1]}_{args[2]}.png',
                otherwise return random image.

        Return:
            (np.ndarray of (733, 1200, 3)) raw image data, where the alpha
                channel is discarded.
            (tuple of 3) digit, xrot, yrot
        '''
        # get raw image name
        if len(args) == 3:
            discriptors = list(map(str, args))
            img = self.imgread(*discriptors, grayscale=self.grayscale)

        else:
            img = None
            while img is None:
                discriptors = list(map(
                    lambda pool: random.sample(pool, 1)[0],
                    [
                        self.avail_digits,
                        self.legal_rots['x'],
                        self.legal_rots['y']
                    ]
                ))
                img = self.imgread(*discriptors, grayscale=self.grayscale)
        img = T.resize(img, (224, 224), anti_aliasing=True)
        return ( img, tuple(map(int, discriptors)) )


    def __len__(self):
        return self.len

    def sample(self):
        '''
        random sample one image

        Return:
            (np.ndarray) image data
            (tuple) digit, xrot, yrot
        '''
        return self.__getitem__()

    def sample_batch(self, batch_size):
        '''
        Return:
            (list of np.ndarray) batch of image
            (list of tuple) labels for this batch
        '''
        # data = [ self.sample() for _ in range(batch_size) ]
        data = []
        for st in range(batch_size):
            # print('collecting data: {:.2f}% ({} of {})'
            #       .format(st * 100.0 / batch_size, st, batch_size),
            #       end='\r')
            data.append(self.sample())
        input_data = []
        truth_labels = []
        for image, label in data:
            input_data.append(image)
            truth_labels.append(label)
        # adapt data to batch-first shape
        # from: [ batch_size, height, width, num_channels ]
        # to: [ batch_size, num_channels, height, width ]
        input_data = torch.tensor(input_data, dtype=torch.float32, device=self.device)
        input_data = torch.transpose(input_data, 3, 2)
        input_data = torch.transpose(input_data, 2, 1)
        return ( input_data, truth_labels )
