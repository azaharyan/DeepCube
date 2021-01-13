import numpy as np
import pycuber as pc

corner_colours = [
    {'white', 'orange', 'blue'},
    {'white', 'orange', 'green'},
    {'white', 'red', 'green'},
    {'white', 'red', 'blue'},
    {'yellow', 'orange', 'blue'},
    {'yellow', 'orange', 'green'},
    {'yellow', 'red', 'green'},
    {'yellow', 'red', 'blue'}
]

corner_colour_map = {
    'white': 0,
    'yellow': 0,
    'blue': 1,
    'green': 1,
    'orange': 2,
    'red': 2
}

edge_colours = [
    {'white', 'blue'},
    {'white', 'orange'},
    {'white', 'green'},
    {'white', 'red'},
    {'yellow', 'blue'},
    {'yellow', 'orange'},
    {'yellow', 'green'},
    {'yellow', 'red'},
    {'orange', 'blue'},
    {'blue', 'red'},
    {'green', 'red'},
    {'orange', 'green'},
]

edge_colour_map = {
    'white': 0,
    'yellow': 0,
    'blue': 0,
    'green': 1,
    'orange': 1,
    'red': 1
}


class CubeEnv:
    def __init__(self, number_of_turns=0, seed=None, cube=None):
        assert number_of_turns == 0 or cube is None  # can't provide both at the same time

        if cube is None:
            self.cube = pc.Cube()
        else:
            self.cube = cube

        self.__solved = self.cube.copy() if cube is None else pc.Cube()

        self.action_space = ["F", "B", "L", "R", "U", "D", "F'", "B'", "L'", "R'", "U'", "D'"]
        self.current_step = 0

        if seed is not None:
            self.__set_seed(seed)

        if cube is None:
            self.reset(number_of_turns)

    def step(self, action):
        old_state = self.cube.copy()
        self.cube.perform_step(self.action_space[action])

        self.current_step += 1

        return old_state, self.reward()

    def reset(self, number_of_turns=0):
        self.cube = pc.Cube()
        self.scramble(number_of_turns)
        self.current_step = 0
        return self.cube

    def render(self):
        print(self.cube)

    def sample(self, policy):
        return policy(range(len(self.action_space)))

    def is_solved(self):
        return self.__solved == self.cube

    def scramble(self, number_of_turns=1):
        formula = ""
        i = 0
        while i < number_of_turns:
            i += 1
            formula += np.random.choice(self.action_space) + " "
        formula = formula.rstrip()
        self.cube(formula)

    def get_one_hot_state(self):
        def comp(x):
            res = list(x.facings.keys())
            res.sort()
            return res
        state = np.zeros((20, 24))
        i = 0
        corners = list(self.cube.select_type('corner'))
        corners.sort(key=comp)
        edges = list(self.cube.select_type('edge'))
        edges.sort(key=comp)
        for corner in corners:
            keys = list(corner.facings.keys())
            keys.sort()
            state[i][3*corner_colours.index(set(map(lambda x: x.colour, corner.facings.values()))) + corner_colour_map[corner.facings[keys[0]].colour]] = 1
            i += 1
        for edge in edges:
            keys = list(edge.facings.keys())
            keys.sort()
            state[i][2*edge_colours.index(set(map(lambda key: edge.facings[key].colour, keys))) + edge_colour_map[edge.facings[keys[0]].colour]] = 1
            i += 1
        return state

    def get_action_space(self):
        return self.action_space

    def get_direct_children_if_not_solved(self):
        direct_children = []
        if not self.is_solved():
            for i in range(len(self.get_action_space())):
                cube = CubeEnv(cube=self.cube.copy())  # TODO: is shallow copy enough?
                cube.step(i)
                direct_children.append(cube)

        return direct_children

    def reward(self):
        return 1 if self.is_solved() else -1

    def __set_seed(self, seed=0):
        np.random.seed(seed)

    def set_state(self, state):
        self.cube = state.copy()

    def __eq__(self, other):
        if not isinstance(other, CubeEnv):
            return NotImplemented
        else:
            return self.cube == other.cube

    def __hash__(self):
        return self.cube.__repr__().__hash__()
