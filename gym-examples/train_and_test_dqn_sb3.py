import gymnasium as gym
import gym_examples
from gym_examples.envs.phil_env import PhilEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from DQN.wrappers import *

env = gym.make("gym_examples/PhilEnv-v1", render_mode="human")
# Frame-stacking with 4 frames using custom gym wrapper
# env = FrameStack(env, 4)

# Instantiate the agent
model = DQN("CnnPolicy", 
            env, 
            buffer_size = 80000, 
            learning_starts = 4000, 
            verbose=1, 
            tensorboard_log = './dqn_tensorboard_log/',
            exploration_fraction = 0.5) # buffer size is 10x smaller than default because my mac doesn't have enough memory

# Train the agent and display a progress bar
model.learn(total_timesteps=int(60000), progress_bar=True)
# Save the agent
model.save("DQN_test")

# del model  # delete trained model to demonstrate loading

# model = DQN.load("DQN_test", env=env)

# #mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# # Enjoy trained agent
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     print(f'Action: {action}')
#     obs, rewards, dones, info = vec_env.step(action) # sb3 does not support terminated and truncated
#     print(f'Observation is {obs} \n'
#           f'Reward is {rewards} \n'
#           f'Done is {dones} \n'
#           f'Distance is {info}')
#     vec_env.render("human") 
