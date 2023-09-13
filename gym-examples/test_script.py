import gymnasium as gym
import gym_examples

env = gym.make('gym_examples/PhilEnv-v1')
observation, info = env.reset()
for i in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f'Observation is {observation} \n'
          f'Distance is {info} \n'
          f'Reward is {reward}')
    if terminated or truncated:
        print(f'Number of timesteps: {i}')
        break
env.close()