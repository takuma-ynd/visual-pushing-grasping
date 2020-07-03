import gym
import numpy as np
from gym import spaces
# from gym.utils import seeding
from phys_intuition_env import PhysIntuitionEnv

class PhysActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.action_space = spaces.Discrete(16 * 224 * 224)
        # spaces.Box(low=np.array([0, 0, 0], dtype=int), high=np.array([16, 224 - 1, 224 - 1], dtype=int), dtype=int)

    def action(self, action):
        best_pix_index = np.unravel_index(action, (16, 224, 224))
        return (0, best_pix_index)  # always PUSH

    def reverse_action(self, action):
        _, act = action
        assert len(act) == 3
        return np.ravel_multi_index(act, (16, 224, 224))
