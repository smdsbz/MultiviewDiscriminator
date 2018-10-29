# -*- coding: utf-8 -*-

import os, time

from matplotlib import image as mpimage

import numpy as np


''' Configurations '''

DATAROOTDIR     = '../download/'

SUMMARYDIR      = './summary/'

MODELDIR        = './save/'
MODELNAME       = 'digit_only_vgg16'
MODELSAVEPATH   = os.path.join(MODELDIR, MODELNAME + '.param')


''' I/O Helper Functions '''

avail_digits = [ d for d in range(10) ]             # 0, ..., 9
legal_xrots = [ an for an in range(0, 360, 10) ]    # 0, ..., 350
legal_yrots = [ an for an in range(90, -90, -10) ]  # -80, ..., 90

gen_filename = (
    lambda dig, xrot, yrot:
        '字母数字符号大全_pic{}_{}_{}.png'
        .format(*list(map(str, [dig, xrot, yrot])))
)

to_grayscale = (
    lambda img:
        np.sum(img[:, :, :3], axis=2) / 3.0
)

def digit_image_read(digit, xrot, yrot, grayscale=True):
    # img = mpimage.imread('../download/'
    #                      + gen_filename(digit, xrot, yrot))
    img = mpimage.imread(os.path.join(
        DATAROOTDIR,
        gen_filename(digit, xrot, yrot)
    ))
    img = img[:, :, :3]  # discard alpha channel
    if grayscale:
        img = to_grayscale(img)
    return img


''' Model Setup '''

# Environment Setup

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# Model Setup

from models import DiscriminatorNet
model = DiscriminatorNet(pretrained=True, device=device).to(device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-6)

which_digit_crit = nn.CrossEntropyLoss()
what_oritent_crit = nn.MSELoss()

# Data Setup

from datautils import DigitDataset
dataloader = DigitDataset(DATAROOTDIR, grayscale=False, device=device)

# Build Pipeline

def test(sample_size=10):
    data, labels = dataloader.sample_batch(sample_size)
    data = torch.tensor(data, dtype=torch.float32, device=device)
    labels = torch.tensor(labels, dtype=torch.float32, device=device)
    digit_labels = labels[:, 0].to(dtype=torch.long)
    digit_pred_score, _ = model(data)
    digit_pred = torch.argmax(digit_pred_score, dim=1)
    acc = (digit_pred == digit_labels).sum().item() / sample_size
    return acc

from itertools import count

def train(
        batch_size=16,
        num_episodes=None):

    writer = SummaryWriter(log_dir=SUMMARYDIR, comment=MODELNAME)

    def train_step(episode_index=None):
        torch.cuda.empty_cache()
        # get network prediction
        data, labels = dataloader.sample_batch(batch_size)  # ( [ img_arr ], [ digit, xrot, yrot ]  )
        # convert data to `torch.tensor`s
        data = torch.tensor(data, dtype=torch.float32, device=device)
        labels = torch.tensor(labels, dtype=torch.float32, device=device)
        digit_labels = labels[:, 0].to(dtype=torch.long)    # labels are index encoded
        orient_labels = labels[:, 1:]
        # make prediction
        digit_pred_score, orient_pred = model(data)
        # calculate loss
        digit_loss = which_digit_crit(digit_pred_score, digit_labels)
        orient_loss = 1e-5 * what_oritent_crit(orient_pred, orient_labels)
        surrogate_loss = digit_loss + 1.0 * orient_loss     # TODO: tune this \tau !!!
        # backward prop and update
        optimizer.zero_grad()
        surrogate_loss.backward()
        optimizer.step()
        # record training progress
        if episode_index:
            writer.add_scalar('train/digit_loss', digit_loss.item(), episode_index)
            writer.add_scalar('train/orient_loss', orient_loss.item(), episode_index)
            writer.add_scalar('train/surrogate_loss', surrogate_loss.item(), episode_index)

    episode_iter = (count() if not num_episodes else range(num_episodes))

    for ep in episode_iter:
        train_step(episode_index=ep)
        if ep % 10 == 0:
            writer.add_scalar('test/test_acc', test(40), ep)


if __name__ == '__main__':

    train(num_episodes=None)
