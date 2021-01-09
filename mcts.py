import math
import random
from collections import deque

import numpy as np

loss_constant = 100
exploration_constant = 100


class MCTS:

    def __init__(self, model):
        self.model = model
        self.children = dict()
        self.score = dict()
        self.number_of_visits = dict()
        self.probabilities = dict()
        self.loss = dict()

    def train(self, state):
        path_to_leaf, actions_to_leaf, leaf = self.traverse(state)
        reward = self.expand(leaf)
        self.backpropagate(path_to_leaf, actions_to_leaf, reward)

        i = 0
        for s in self.children[leaf]:
            if s.is_solved():
                actions_to_leaf.append(i)
                actions_to_leaf_bfs = self.bfs(state)
                return actions_to_leaf, actions_to_leaf_bfs
            i += 1

        return None, None

    def traverse(self, state):
        path_to_leaf = []
        actions_to_leaf = []
        current = state
        while True:
            if current not in self.children or not self.children[current]:
                return path_to_leaf, actions_to_leaf, current
            else:
                if sum(self.number_of_visits[current]) == 0:
                    action_index = random.randint(0, len(current.get_action_space()) - 1)
                else:
                    action_index = self.get_most_promising_action_index(current)

                path_to_leaf.append(current)
                actions_to_leaf.append(action_index)
                current = self.children[current][action_index]

    def expand(self, state):
        value, policy = self.model.predict(np.array(state.get_one_hot_state()).flatten()[None, :])

        self.probabilities[state] = policy[0]
        self.loss[state] = [0] * len(state.get_action_space())
        self.score[state] = [0] * len(state.get_action_space())
        self.number_of_visits[state] = [0] * len(state.get_action_space())

        self.children[state] = state.get_direct_children_if_not_solved()

        return value[0][0]

    def backpropagate(self, path_to_leaf, actions_to_leaf, reward):
        for state_to_leaf, action_to_leaf in zip(path_to_leaf, actions_to_leaf):
            self.score[state_to_leaf][action_to_leaf] = max(self.score[state_to_leaf][action_to_leaf], reward)
            self.number_of_visits[state_to_leaf][action_to_leaf] += 1
            self.loss[state_to_leaf][action_to_leaf] += loss_constant

    def get_most_promising_action_index(self, state):
        action_space_len = len(state.get_action_space())

        state_all_actions_number_of_visits = sum(self.number_of_visits[state])
        u_plus_w_a = [0] * action_space_len
        for i in range(action_space_len):
            u_st_a = exploration_constant \
                     * self.probabilities[state][i] \
                     * (math.sqrt(state_all_actions_number_of_visits) / (1 + self.number_of_visits[state][i]))
            u_plus_w_a[i] = u_st_a + self.score[state][i] - self.loss[state][i]

        return max(range(len(u_plus_w_a)), key=u_plus_w_a.__getitem__)

    """
    Leaving here as I wrote it for testing to make sure bfs works
    
    def expand_levels(self, state, current_depth, max_depth):
        if current_depth > max_depth:
            return

        direct_children = state.get_direct_children_if_not_solved()
        self.children[state] = direct_children
        for direct_child in direct_children:
            self.expand_levels(direct_child, current_depth + 1, max_depth)
    """

    def bfs(self, state):
        """
        Uncomment and invoke method directly in main with matching number_of_turns and max_depth
        to make sure bfs works

        self.children = dict()
        self.expand_levels(state, 0, 2)
        """

        visited = {state}
        solved = None
        state_to_parent_and_index_from_parent = dict()
        state_to_parent_and_index_from_parent[state] = (None, None)

        queue = deque()
        queue.append(state)
        while len(queue) != 0:
            current = queue.popleft()
            if current.is_solved():
                solved = current
                break

            if current not in self.children:  # in MCTS not all branches are visited
                continue

            i = 0
            for current_child in self.children[current]:
                if current_child not in visited:
                    queue.append(current_child)
                    state_to_parent_and_index_from_parent[current_child] = (current, i)
                    visited.add(current_child)

                i += 1

        if solved is None:
            return None

        current = solved
        reversed_actions_to_leaf = []
        while True:
            pair = state_to_parent_and_index_from_parent[current]
            current = pair[0]
            if current is None:
                break

            reversed_actions_to_leaf.append(pair[1])

        reversed_actions_to_leaf.reverse()
        return reversed_actions_to_leaf
