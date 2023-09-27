import numpy as np
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from gym import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper
import cv2
import sys
from collections import deque

class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames

    def __array__(self, dtype = None):
        output = np.concatenate(self._frames, axis = 0)
        if dtype is not None:
            output = output.astype(dtype)
        return output 
    
    def __len__(self):
        return len(self._frames)
    
    def __getitem__(self, i):
        return self._frames[i]

class FrameStack(gym.Wrapper):
    """
    Stack every 4 last frames so that obs space shape is (4, 84, 84) 
    """
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque([], maxlen = num_stack) # create rolling buffer
        env_shape = env.observation_space.shape # will return (1, 84, 84)
        self.observation_space = spaces.Box(low=0, high = 255, shape = (env_shape[0] * self.num_stack, env_shape[1], env_shape[2]), dtype = np.uint8) # change obs shape into (4, 84, 84)

    def reset(self, seed = None):

        if seed is not None:
            self.env.seed(seed)

        obs, info = self.env.reset()
      #  obs = np.array(obs).transpose(2,0,1)
        for i in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs()
    
    def step(self, action):
        obs, reward, terminated, _, info = self.env.step(action)
      #  obs = np.array(obs).transpose(2,0,1)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self):
        assert len(self.frames) == self.num_stack
        return LazyFrames(list(self.frames))
    
    
 