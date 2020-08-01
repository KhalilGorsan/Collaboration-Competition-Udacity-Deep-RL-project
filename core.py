from pathlib import Path

import numpy as np
from unityagents import UnityEnvironment


class TennisWrapper:
    """Tennis Unity Environment Wrapper.
    """

    def __init__(self, file_name: Path):
        self.env = UnityEnvironment(file_name)
        # Get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

    def reset(self):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        state = env_info.vector_observations[0]
        return state

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return next_state, reward, done

    def close(self):
        self.env.close()

    @property
    def action_size(self):
        return self.brain.vector_action_space_size

    @property
    def observation_size(self):
        return self.brain.vector_observation_space_size