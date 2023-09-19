import math
import time
import os
import numpy as np
import random
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import cv2
import sys
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete):
        super(DQN, self).__init__():

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 8, stride = 4)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear()

        # Note: Height out = (Hin - kernel_size + 2*padding)/ stride + 1
    #Height and width out = Hin/kernel_size

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size(0), -1) # flatten
        out = self.fc1(out)
        out = self.relu4(out)
        out = self.fc2(out)

        return out

























