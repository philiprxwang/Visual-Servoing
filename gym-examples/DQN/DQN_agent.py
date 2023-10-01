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
from DQN.DQN_replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete,
                 replay_buffer: ReplayBuffer,
                 lr,
                 batch_size,
                 gamma,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')):
        
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

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = np.array(states) / 255.0
        next_states = np.array(next_states) / 255.0
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).short().to(device)
        
        with torch.no_grad(): # code below not tracked for gradient computation 
            next_q_vals = self.target_network(next_states)
            next_q_vals_max, _ = next_q_vals.max(1) # Select max value in each state/row
            # print(f'1-dones is {1-dones}')
            # print(f'Self.gamma is {self.gamma}')
            # print(f'Next_q_values max is {next_q_vals_max}')
            # print(f'Rewards is {rewards}')
            # x = (1 -dones) * self.gamma * next_q_vals_max
            # print(f'x is {x}')

            target_q_vals = rewards + (1-dones) * self.gamma * next_q_vals_max 

        input_q_values = self.policy_network(states)
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        criterion = nn.SmoothL1Loss()
        loss = criterion(input_q_values, target_q_vals)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        del states
        del next_states

        return loss.item()
    
    def update_target_network(self):

        self.target_network.load_state_dict(self.policy_network.state_dict())
    
    def act(self, state):

        device = self.device

        state = np.array(state) / 255.0 # normalize and then convert to float before passing to DQN
      #  print(f'State shape bef squeezing: {state.shape}')
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) # (4, 84, 84) --> (1, 4, 84, 84)
       # print(f'State shape after squeezing: {state.shape}')

        with torch.no_grad():
            q_values = self.policy_network(state)
            _, action = q_values.max(1)
            return action.item() # return as standard number 
