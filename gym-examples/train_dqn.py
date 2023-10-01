import numpy as np
import random
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import gym_examples
import cv2
import sys
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from DQN.DQN_model import DQN
from DQN.DQN_replay_buffer import ReplayBuffer
from DQN.DQN_agent import DQNAgent
from DQN.wrappers import *
import argparse
from tqdm import tqdm
import contextlib

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'DQN PhilEnv')
    parser.add_argument('--load-checkpoint-file', type = str, default = None, help = 'Where checkpoint file should be loaded from')

    args = parser.parse_args()

    if (args.load_checkpoint_file):
        epsilon_start = 0.01 # minimal exploration if file exists
    else:
        epsilon_start = 1

    hyper_params = {
        'seed': 42,
        'env': 'gym_examples/PhilEnv-v1',
        'replay_buffer_size': 20000,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'total_steps': 60000,
        'batch_size': 32,
        'steps_before_learning': 1000,
        'target_network_update_freq': 500,
        'eps_start': epsilon_start,
        'eps_end': 0.01,
        'eps_fraction': 0.5,
        'print_freq': 10
    }

    np.random.seed(hyper_params['seed'])
    random.seed(hyper_params['seed'])

    writer = SummaryWriter("DQN_PhilEnv")

    env = gym.make('gym_examples/PhilEnv-v1')
 #   env.seed(hyper_params['seed'])
    env = FrameStack(env, 4)

    replay_buffer = ReplayBuffer(hyper_params['replay_buffer_size'])

    agent = DQNAgent(
        observation_space = env.observation_space,
        action_space = env.action_space,
        replay_buffer = replay_buffer,
        lr = hyper_params['learning_rate'],
        batch_size = hyper_params['batch_size'],
        gamma = hyper_params['gamma'],
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    )

    if (args.load_checkpoint_file):
        print(f"Loading existing policy: {args.load_checkpoint_file}")
        agent.policy_network.load_state_dict(torch.load(args.load_checkpoint_file))

    epsilon_annealing_timesteps = hyper_params['eps_fraction'] * float(hyper_params['total_steps'])
    episode_rewards = [0.0]

   # state= env.reset(seed = hyper_params['seed'])
    state = env.reset()
   # print(f'Length of state: {len(state)}')

    
    class DummyFile(object):
        '''
        DummyFile class used to change printing to sys.stdout to ensure tdqm progress bar stays at bottom
        https://stackoverflow.com/questions/36986929/redirect-print-command-in-python-script-through-tqdm-write/37243211#37243211
        '''
        file = None
        def __init__(self, file):
            self.file = file

        def write(self, x):
            # Avoid print() second call (useless \n)
            if len(x.rstrip()) > 0:
                tqdm.write(x, file=self.file)

    @contextlib.contextmanager
    def nostdout():
        save_stdout = sys.stdout
        sys.stdout = DummyFile(sys.stdout)
        yield
        sys.stdout = save_stdout

    for i in tqdm(range(hyper_params['total_steps']), file = sys.stdout):
        with nostdout():
            fraction = min(1.0, float(i) / epsilon_annealing_timesteps) # current timestep as a fraction of total annealing timesteps
            eps_thresh = hyper_params['eps_start'] + fraction * (hyper_params['eps_end'] - hyper_params['eps_start']) # when fraction==1, eps_thresh = eps_end

            sample = random.random()

            if sample > eps_thresh:
                action = agent.act(state) # Act greedily
            else:
                action = env.action_space.sample() # Exploration

            next_state, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                done = 1
            else: 
                done = 0

            agent.memory.add(state, action, reward, done, next_state) # add into Replay Buffer
            # not sure that if done, next_state should be None - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
            state = next_state

            episode_rewards[-1] += reward

            if done:
                state = env.reset()
                episode_rewards.append(0.0) # new item for new episode 

            if i < hyper_params['steps_before_learning']:
                loss = 0 # set loss to 0 first because loss undefined when i < hyper_params['steps_before_learning']

            if i > hyper_params['steps_before_learning']:
                loss = agent.optimise_td_loss()
            
            if i > hyper_params['steps_before_learning'] and i % hyper_params['target_network_update_freq'] == 0: # update target network
                agent.update_target_network()

            num_episodes = len(episode_rewards)
            
            if done:
                print(f'Number of eps: {num_episodes}')

            if i % hyper_params['print_freq'] == 0:
                writer.add_scalar('Mean reward', round(np.mean(episode_rewards[-11:-1]), 1), i)
                writer.add_scalar('Loss', loss, i)
                writer.add_scalar('Epsilon', eps_thresh*100, i)
                writer.close() 

            if done and num_episodes % hyper_params['print_freq'] == 0: 
                mean_reward = round(np.mean(episode_rewards[-11:-1]), 1)
                print('***************************************')
                print(f'Timesteps: {i}')
                print(f'Number of episodes: {num_episodes}')
                print(f'% time exploring: {eps_thresh*100}')
                print(f'Mean reward: {mean_reward}')
                
                torch.save(agent.policy_network.state_dict(), f'checkpoint.pth')
                np.savetxt('rewards_per_episode.csv', episode_rewards, delimiter = ',', fmt='%1.3f')
                





