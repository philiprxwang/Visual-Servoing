import gymnasium as gym
import gym_examples

env = gym.make('gym_examples/PhilEnv-v0')
observation, info = env.reset()
for i in range(1000):
    env.render()
    print(observation)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f'Observation is {observation}'
          f'Distance is {info}'
          f'Reward is {reward}')
    if terminated or truncated:
        print(f'Number of timesteps: {i}')
        break
env.close()