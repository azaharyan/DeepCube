from cubeEnv import CubeEnv
from mcts import MCTS
from model import buildModel, compile_model
from tensorflow import keras

if __name__ == "__main__":

    print(keras.__version__)
    model = keras.models.load_model('saved_model')

    #for new model
    #model = buildModel(20 * 24)
    #compile_model(model, 0.001)

    mcts = MCTS(model)
    cube = CubeEnv(number_of_turns=3)

    cube.render()

    pair = (None, None)
    while not pair[0]:
        pair = mcts.train(cube)

    action_space = cube.get_action_space()
    print("naive: ")
    print([action_space[action_index] for action_index in pair[0]])
    print("bfs: ")
    print([action_space[action_index] for action_index in pair[1]])
