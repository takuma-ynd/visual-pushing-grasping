import random
# from mypytools.misc import print_blue
from chainerrl.explorer import Explorer
from chainerrl.explorers import Boltzmann

class PhysExplorer(Explorer):
    def __init__(self, env):
        super().__init__()
        self._env = env
        self._explorers = Boltzmann()

    def select_action(self, t, greedy_action_func, action_value=None):
        if  t < 1000 and random.random() < 0.7:
            # print_blue('Bootstrapped PUSH at t={}'.format(t))
            action = self._env.get_heuristic_action(0, self._env.shared_obs.valid_depth_heightmap)
            action = self._env.reverse_action(action)  # convert tuple of (3, ) --> int
        else:
            # print_blue('Boltzmann PUSH at t={}'.format(t))
            action = self._explorers.select_action(t, greedy_action_func, action_value)

        return action
