import pycuber as pc
import numpy as np

class CubeEnv():
    def __init__(self, number_of_turns = 0, seed = 0):
        self.cube = pc.Cube()
        self.__solved = self.cube.copy()

        self.action_space = ["F", "B", "L", "R", "U", "D", "F'", "B'", "L'", "R'", "U'", "D'"]

        self.__set_seed(seed)
        self.reset(number_of_turns)

    def step(self, action):
        old_state = self.cube.copy()
        self.cube.perform_step(self.action_space[action])

        self.current_step += 1
        reward = 1 if self.is_solved() else -1

        return old_state, reward

    def reset(self, number_of_turns = 0):
        self.current_step = 0
        self.cube = pc.Cube()
        self.scramble(number_of_turns)
        return self.cube

    def render(self):
        print(self.cube)

    def sample(self, policy):
        return policy(range(len(self.action_space)))

    def is_solved(self):
        return self.__solved == self.cube

    def scramble(self, number_of_turns = 1):
        formula = ""
        i = 0
        while(i<number_of_turns):
            i += 1
            formula += np.random.choice(self.action_space) + " "
        formula.rstrip()
        self.cube(formula)

    def get_action_space(self):
        return self.action_space

    def __set_seed(self,seed=0):
        np.random.seed(seed)
