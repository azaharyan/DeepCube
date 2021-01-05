import math
import numpy as np

from cubeEnv import CubeEnv

exploration_constant = 100


class MCTS:

    def __init__(self, model):
        self.model = model
        self.children = dict()
        self.score = dict()
        self.number_of_visits = dict()
        self.probabilities = dict()

    def train(self, state):
        path_to_leaf, actions_to_leaf, leaf = self.traverse(state)
        reward = self.expand(leaf)
        self.rollout(leaf)
        self.backpropagate(path_to_leaf, actions_to_leaf, reward)

        solved = []
        found = False
        for s in self.children[leaf]:
            if s.is_solved():
                solved.append(s)
                found = True

        if found:
            return solved, path_to_leaf
        else:
            return None, None

    def traverse(self, state):
        path_to_leaf = []
        actions_to_leaf = []
        current = state
        while True:
            if current not in self.children or not self.children[current]:
                return path_to_leaf, actions_to_leaf, current
            else:
                not_explored_children = [child for child in self.children[current] if child not in self.children.keys()]
                if not_explored_children:
                    first_not_explored_child = not_explored_children[0]
                    path_to_leaf.append(current)
                    actions_to_leaf.append(self.children[current].index(first_not_explored_child))
                    return path_to_leaf, actions_to_leaf, first_not_explored_child
                else:
                    action_index = self.get_most_promising_action_index(current)
                    path_to_leaf.append(current)
                    actions_to_leaf.append(action_index)
                    current = self.children[state][action_index]

    def expand(self, state):
        value, policy = self.model.predict(np.array(state.get_one_hot_state()).flatten()[None, :])
        if state in self.children:
            return value[0][0]

        self.initialize_state_data(state, policy)
        self.children[state] = state.get_direct_children_if_not_solved()

        for child in self.children[state]:
            self.initialize_state_data(child)

        return value[0][0]

    def initialize_state_data(self, state, policy=None):  # allow for providing policy to avoid predicting twice
        if state not in self.score:  # doesn't matter which of score, probabilities or number_of_visits is used here
            if policy is None:
                _, policy = self.model.predict(np.array(state.get_one_hot_state()).flatten()[None, :])
            self.score[state] = {}
            self.probabilities[state] = {}
            self.number_of_visits[state] = {}
            for i in range(len(state.get_action_space())):
                self.score[state][i] = 0
                self.probabilities[state][i] = policy[0][i]
                self.number_of_visits[state][i] = 0

    def rollout(self, state):
        """
        During this step usually child states are generated until final state is reached
        This is not feasible in our situation due to the nature of the problem we are solving
        """
        pass

    def backpropagate(self, path_to_leaf, actions_to_leaf, reward):
        for state_to_leaf, action_to_leaf in zip(path_to_leaf, actions_to_leaf):
            self.score[state_to_leaf][action_to_leaf] = max(self.score[state_to_leaf][action_to_leaf], reward)
            self.number_of_visits[state_to_leaf][action_to_leaf] += 1

    def get_most_promising_action_index(self, state):
        state_all_actions_number_of_visits = 0
        action_space_len = len(state.get_action_space())
        for i in range(action_space_len):
            state_all_actions_number_of_visits += self.number_of_visits[state][i]

        u_plus_w_a = [0] * action_space_len
        for i in range(action_space_len):
            u_st_a = exploration_constant\
                   * self.probabilities[state][i]\
                   * (math.sqrt(state_all_actions_number_of_visits) / (1 + self.number_of_visits[state][i]))
            u_plus_w_a[i] = u_st_a + self.score[state][i]

        return max(range(len(u_plus_w_a)), key=u_plus_w_a.__getitem__)

    def play(self, state):
        if state.is_solved():
            raise Exception

        cube = CubeEnv(cube=state.cube.copy())  # TODO: is shallow copy enough?
        if state not in self.children:
            cube.scramble()
        else:
            cube.step(self.get_most_promising_action_index(state))

        return cube
