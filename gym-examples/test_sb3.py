import gymnasium as gym
import gym_examples
from gym_examples.envs.phil_env import PhilEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("gym_examples/PhilEnv-v1", render_mode="human")
model = A2C.load("A2C_test", env=env)

#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    print(f'Action: {action}')
    obs, rewards, dones, info = vec_env.step(action) # sb3 does not support terminated and truncated
    print(f'Reward is {rewards} \n'
          f'Done is {dones} \n'
          f'Distance is {info}')
    vec_env.render("human") 