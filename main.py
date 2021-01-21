import time

from cubeEnv import CubeEnv
from mcts import MCTS
from model import buildModel, compile_model
from tensorflow import keras

if __name__ == "__main__":

    print(keras.__version__)
    model = keras.models.load_model("last_hope.hdf5")

    #for new model
    #model = buildModel(20 * 24)
    #compile_model(model, 0.001)

    mcts = MCTS(model)
    cube = CubeEnv(number_of_turns=3)

    cube.render()

    start_time = time.time()
    naive_action_indexes = None
    while not naive_action_indexes:
        naive_action_indexes = mcts.train(cube)
    end_time = time.time()

    action_space = cube.get_action_space()
    print("{}s, naive: ".format(end_time - start_time))
    print([action_space[action_index] for action_index in naive_action_indexes])

    start_time = time.time()
    bfs_action_indexes = mcts.bfs(cube)
    end_time = time.time()

    print("{}s, bfs afterwards: ".format(end_time - start_time))
    print([action_space[action_index] for action_index in bfs_action_indexes])
