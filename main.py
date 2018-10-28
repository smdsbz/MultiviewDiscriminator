# -*- coding: utf-8 -*-

import torch
import torchvision

from models import InceptionNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
