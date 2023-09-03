from gymnasium.envs.registration import register

register(
    id='gym_examples/PhilEnv-v0',
    entry_point='gym_examples.envs:PhilEnv',
    max_episode_steps=1000
)

# After registration, our custom PhilEnv environment can be created with env = gymnasium.make('PhilEnv-v0').