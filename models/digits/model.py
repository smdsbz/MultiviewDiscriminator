# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DiscriminatorNet(nn.Module):
    ''' Discriminator Model'''

    def __init__(self, pretrained=False, device=device):
        '''
        Args:
            pretrained (bool): use pretrained model from PyTorch.org
        '''
        super(DiscriminatorNet, self).__init__()
        self.device = device
        # using InceptionV3 for feature extraction
        # self.feature_extraction = torchvision.models.inception_v3(pretrained=pretrained)
        self.feature_extraction = torchvision.models.vgg16(pretrained=pretrained).eval()  # freeze its parameters!!!
        # sub-model for digit classification
        self.which_digit_pred = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096, bias=True), nn.LeakyReLU(),
            nn.Linear(4096, 4096, bias=True), nn.LeakyReLU(),
            nn.Linear(4096, 10)
        )
        # sub-models for orienting
        self.what_orient_pred = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096, bias=True), nn.LeakyReLU(),
            nn.Linear(4096, 4096, bias=True), nn.LeakyReLU(),
            nn.Linear(4096, 2, bias=True)
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
        x = self.feature_extraction.features(x).view(x.shape[0], -1)  # only use feature extraction layers of pretrained net
        x = x.detach()      # NOTE: leave feature extraction net undisturbed
        which_digit = self.which_digit_pred(x)
        what_orient = self.what_orient_pred(x)
        what_orient[:, 0].clamp_(0, 360)
        what_orient[:, 1].clamp_(-80, 90)
        return ( which_digit, what_orient )
