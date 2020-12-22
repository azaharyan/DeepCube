import pycuber as pc
import numpy as np

class CubeEnv():
    def __init__(self, number_of_turns = 0, seed = 0):
        self.cube = pc.Cube()
        self.__solved = self.cube.copy()

        self.action_space = ["F", "B", "L", "R", "U", "D", "F'", "B'", "L'", "R'", "U'", "D'"]

        self.__set_seed(seed)
        self.reset(number_of_turns)

        self.__corner_colours = [
            {'white', 'orange', 'blue'},
            {'white', 'orange',  'green'},
            {'white', 'red',  'green'},
            {'white', 'red', 'blue'},
            {'yellow', 'orange', 'blue'},
            {'yellow', 'orange',  'green'},
            {'yellow', 'red',  'green'},
            {'yellow', 'red', 'blue'}
        ]

        self.__colour_map = {
            'white': 0,
            'yellow': 0,
            'blue': 1,
            'green':1,
            'orange': 2,
            'red': 2
        }

        self.__edge_colours = [
            ['white', 'blue'],
            ['white', 'orange'],
            ['white', 'green'],
            ['white', 'red'],
            ['yellow', 'blue'],
            ['yellow', 'orange'],
            ['yellow', 'green'],
            ['yellow', 'red'],
            ['blue', 'white'],
            ['orange', 'white'],
            ['green', 'white'],
            ['red','white'],
            ['blue','yellow'],
            ['orange', 'yellow'],
            ['green', 'yellow'],
            ['red', 'yellow'],
            ['orange', 'blue'],
            ['blue', 'orange'],
            ['green', 'orange'],
            ['orange', 'green'],
            ['red', 'blue'],
            ['blue', 'red'],
            ['green', 'red'],
            ['red', 'green']
        ]

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

    def get_one_hot_state(self):
        state = np.zeros((20,24))
        i = 0
        corners = list(self.cube.select_type('corner'))
        corners.sort(key=lambda x: hash(x))
        edges = list(self.cube.select_type('edge'))
        edges.sort(key=lambda x: hash(x))
        for corner in corners:
            keys = list(corner.facings.keys())
            keys.sort()
            state[i][3*self.__corner_colours.index(set(map(lambda x: x.colour, corner.facings.values()))) + self.__colour_map[corner.facings[keys[0]].colour]] = 1
            i += 1
        for edge in edges:
            keys = list(edge.facings.keys())
            keys.sort()
            state[i][self.__edge_colours.index(list(map(lambda key: edge.facings[key].colour, keys)))] = 1
            i += 1
        return state

    def __set_seed(self,seed=0):
        np.random.seed(seed)
