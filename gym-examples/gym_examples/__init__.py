from gymnasium.envs.registration import register

register(
    id='gym_examples/PhilEnv-v1',
    entry_point='gym_examples.envs:PhilEnv',
    max_episode_steps=100
)