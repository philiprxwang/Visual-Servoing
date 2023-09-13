import gymnasium as gym
import gym_examples
from gym_examples.envs.phil_env import PhilEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# env = gym.make("gym_examples/PhilEnv-v1", render_mode = 'human')
# env.reset()
# check_env(env, warn=True) # check_env closes env after a brief moment
vec_env = make_vec_env(PhilEnv, n_envs = 2)
model = A2C('MlpPolicy', env = vec_env, verbose=1).learn(total_timesteps=2, progress_bar=True)
model.save("A2C_PhilEnv")

del model

model = A2C.load("A2C_PhilEnv")

avg_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    print(f'Action: {action}')
    obs, reward, dones, info = vec_env.step(action) # sb3 does not support terminated and truncated
    print(f'Observation is {obs} \n'
          f'Reward is {reward} \n'
          f'Done is {dones}')
    vec_env.render('human')

