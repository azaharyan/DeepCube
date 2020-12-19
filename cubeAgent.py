import numpy as np
from cubeEnv import CubeEnv

class CubeAgent():
    def __init__(self, number_of_cubes = 3, n_time_steps = 2, seed = 0):
        self.env = [CubeEnv() for i in range(number_of_cubes)]
        self.n_time_steps = n_time_steps
        self.number_of_cubes = number_of_cubes

    def scrabmle_cubes_for_data(self):
        i = 1
        while (i < self.number_of_cubes):
            self.env[i].cube = self.env[i-1].cube.copy()
            self.env[i].scramble(1)
            i += 1

    def take_actions(self, policy):
        i = 0
        j = 0
        while (j < self.number_of_cubes):
            j += 1
            while(i < self.n_time_steps):
                action = self.env[j].sample(policy)
                _, reward = self.env[j].step(action)
                i += 1
                if self.env[j].is_solved():
                    print(f'cube {i} solved')
                    break

    def reset_envs(self, number_of_turns = None):
        if number_of_turns is None:
            number_of_turns = [0 for j in range(self.number_of_cubes)]
        i = 0
        while (i < self.number_of_cubes):
            self.env[i].reset(number_of_turns[i])
            i += 1

    def get_cube(self, index=0):
        return self.env[index].cube