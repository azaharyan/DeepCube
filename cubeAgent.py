import numpy as np
from cubeEnv import CubeEnv

class CubeAgent():
    def __init__(self, n_time_steps = 30, seed = 0):
        self.env = CubeEnv(1, seed)
        self.n_time_steps = n_time_steps

    def take_actions(self, policy):
        i = 0
        while(i < self.n_time_steps):
            action = self.env.sample(policy)
            _, reward = self.env.step(action)
            i += 1
            if self.env.is_solved():
                print("cube solved")
                break
