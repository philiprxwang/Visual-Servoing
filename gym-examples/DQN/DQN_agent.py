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
from DQN.DQN_model import DQN


class DQNAgent:
    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete,
                 replay_buffer: ReplayBuffer,
                 lr,
                 batch_size,
                 gamma,
                 device = torch.device('mps')):
        # MPS NEEDS TO BE EVENTUALLY CHANGED!
        
        self.memory = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy_network = DQN(observation_space, action_space).to(device)
        self.target_network = DQN(observation_space, action_space).to(device)
        self.update_target_network()
        self.target_network.eval()
        self.optimiser = torch.optim.RMSprop(self.policy_network.parameters(), lr = lr)
        self.device = device
    
    def optimise_td_loss(self):

        device = self.device

        states, actions, rewards, next_states, terminated, truncated = self.memory.sample()
        states = torch.from_numpy(states).uint8().to(device)
        actions = torch.from_numpy(actions).uint8().to(device)
        rewards = torch.from_numpy(actions).float64().to(device)
        next_states = torch.from_numpy(next_states).float64().to(device)
        terminated = torch.from_numpy(terminated).float64().to(device)
        truncated = torch.from_numpy(truncated).float64().to(device)
        
        with torch.no_grad(): # code below not tracked for gradient computation 
            next_q_vals = self.target_network(next_states)
            next_q_vals_max = next_q_vals.max(1)
            target_q_vals = rewards + (1-terminated) * self.gamma * next_q_vals_max # TD target - how to add truncated?

        input_q_values = self.policy_network(states)
        intput_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()




