import math
import random
from collections import deque

import numpy as np


class MCTS:

    def __init__(self, model, loss_constant=150, exploration_constant=4):
        self.model = model

        self.children_and_data = dict()

        self.ch_i = 0
        self.p_i = 1
        self.s_i = 2
        self.n_of_v_i = 3
        self.v_l_i = 4

        self.loss_constant = loss_constant
        self.exploration_constant = exploration_constant

    def train(self, state):
        path_to_leaf, actions_to_leaf, leaf = self.traverse(state)
        reward = self.expand(leaf)
        self.backpropagate(path_to_leaf, actions_to_leaf, reward)

        i = 0
        for s in self.children_and_data[leaf][self.ch_i]:
            if s.is_solved():
                actions_to_leaf.append(i)
                return actions_to_leaf
            i += 1

        return None

    def traverse(self, state):
        path_to_leaf = []
        actions_to_leaf = []
        current = state
        while True:
            if current not in self.children_and_data or not self.children_and_data[current][self.ch_i]:
                return path_to_leaf, actions_to_leaf, current
            else:
                if sum(self.children_and_data[current][self.n_of_v_i]) == 0:
                    action_index = random.randint(0, len(current.get_action_space()) - 1)
                else:
                    action_index = self.get_most_promising_action_index(current)

                path_to_leaf.append(current)
                actions_to_leaf.append(action_index)
                self.children_and_data[current][self.v_l_i][action_index] += self.loss_constant

                current = self.children_and_data[current][self.ch_i][action_index]

    def expand(self, state):
        value, policy = self.model.predict(np.array(state.get_one_hot_state()).flatten()[None, :])

        self.children_and_data[state] = (
            state.get_direct_children_if_not_solved(),
            policy[0],
            [0] * len(state.get_action_space()),
            [0] * len(state.get_action_space()),
            [0] * len(state.get_action_space()))

        return value[0][0]

    def backpropagate(self, path_to_leaf, actions_to_leaf, reward):
        for state_to_leaf, action_to_leaf in zip(path_to_leaf, actions_to_leaf):
            self.children_and_data[state_to_leaf][self.s_i][action_to_leaf] = max(
                self.children_and_data[state_to_leaf][self.s_i][action_to_leaf], reward)

            self.children_and_data[state_to_leaf][self.n_of_v_i][action_to_leaf] += 1

    def get_most_promising_action_index(self, state):
        action_space_len = len(state.get_action_space())

        state_all_actions_number_of_visits = sum(self.children_and_data[state][self.n_of_v_i])
        u_plus_w_a = [0] * action_space_len
        for i in range(action_space_len):
            u_st_a = self.exploration_constant \
                     * self.children_and_data[state][self.p_i][i] \
                     * (math.sqrt(state_all_actions_number_of_visits)
                        / (1 + self.children_and_data[state][self.n_of_v_i][i]))
            u_plus_w_a[i] = u_st_a \
                            + self.children_and_data[state][self.s_i][i] \
                            - self.children_and_data[state][self.v_l_i][i]

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

            if current not in self.children_and_data:  # in MCTS not all branches are visited
                continue

            i = 0
            for current_child in self.children_and_data[current][self.ch_i]:
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
