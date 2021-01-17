import time
import numpy as np
from cubeEnv import CubeEnv


class GreedyBestFirstSearch:

    def __init__(self, model, max_time_in_seconds):
        self.model = model
        self.max_time_in_seconds = max_time_in_seconds

    def gbfs(self, state):
        if state.is_solved():
            return []

        state = CubeEnv(cube=state.cube.copy())

        path = []
        start_time = time.time()
        while start_time + self.max_time_in_seconds > time.time():
            _, probabilities = self.model.predict(np.array(state.get_one_hot_state()).flatten()[None, :])
            best_action_index = probabilities[0].argmax()
            state.step(best_action_index)
            path.append(best_action_index)
            if state.is_solved():
                return path

        return None
