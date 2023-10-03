import gymnasium as gym
import gym_examples
from gym_examples.envs.phil_env import PhilEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import argparse

def main():

    parser = argparse.ArgumentParser(description='DQN Train')

    parser.add_argument('--stacking', type = str, default = "True", help = 'Whether or not to use frame stacking - True or False')
    parser.add_argument('--render', type = str, default = "False", help = 'Whether to render or not - True or False')

    args = parser.parse_args()

    render_mode = 'human' if args.render.lower() == 'true' else None

    if args.stacking.lower() == "true":
        # Wrap in SB3 Monitor first to see rewards in DummyVecEnv
        env = Monitor(gym.make("gym_examples/PhilEnv-v1", render_mode = render_mode))
        env = DummyVecEnv([lambda: env])
        # Frame-stacking with 4 frames using SB3 wrapper
        env = VecFrameStack(env, 4)

        # Instantiate the agent
        # ADD TARGET_UPDATE_INTERVAL? Default is 10000
        model = DQN("CnnPolicy", 
                    env, 
                    buffer_size = 20000, # buffer size is smaller in stacking==True because 4 frames take up more memory
                    learning_starts = 1000,
                    verbose=1, 
                    tensorboard_log = './dqn_tensorboard_log/',
                    exploration_fraction = 0.5) # buffer size is 10x smaller than default because my mac doesn't have enough memory

    else:
        env = gym.make("gym_examples/PhilEnv-v1", render_mode=render_mode)

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

if __name__ == "__main__":
    main()