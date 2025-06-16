from gymnasium.envs.registration import register

register(
    id="aipal/IndustrialEnv-v0",
    entry_point="aipal_industrial_dqn.industrial_env:IndustrialAutomationEnv",
    max_episode_steps=300,
)
