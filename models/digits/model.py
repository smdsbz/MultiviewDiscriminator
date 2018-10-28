# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InceptionNet(nn.Module):
    ''' Inception-Based Model'''

    def __init__(self, pretrained=False, device=device):
        '''
        Args:
            pretrained (bool): use pretrained model from PyTorch.org
        '''
        super(InceptionNet, self).__init__()
        self.device = device
        # using InceptionV3 for feature extraction
        self.inception = torchvision.models.inception_v3(pretrained=pretrained)
        # sub-model for digit classification
        self.which_digit_pred = nn.Sequential(
            nn.Linear(1000, 512, bias=True), nn.LeakyReLU(),
            nn.Linear(512, 512, bias=True), nn.LeakyReLU(),
            nn.Linear(512, 10), nn.Softmax()
        )
        # sub-models for orienting
        self.what_orient_pred = nn.Sequential(
            nn.Linear(1000, 512, bias=True), nn.LeakyReLU(),
            nn.Linear(512, 512, bias=True), nn.LeakyReLU(),
            nn.Linear(512, 2, bias=True)
        )

    def forward(self, x):
        '''
        Args:
            x (np.ndarray of (733, 1200)): grayscale image tensor.

        Return:
            which_digit (Tensor of (10,)): output probability values for each
                digit.
            what_orient (Tensor of (2,)): xrot and yrot, respectively.
        '''
        x = self.inception(x)
        which_digit = self.which_digit_pred(x)
        xrot, yrot = self.what_orient_pred(x)
        xrot.clamp_(0, 360)
        yrot.clamp_(-80, 90)
        what_orient = torch.tensor([xrot, yrot])
        return ( which_digit, what_orient )
