from cubeEnv import CubeEnv
from mcts import MCTS
from model import buildModel, compile_model
from tensorflow import keras

if __name__ == "__main__":
    model = keras.models.load_model('saved_model')

    #for new model
    #model = buildModel(20 * 24)
    #compile_model(model, 0.001)

    mcts = MCTS(model)
    cube = CubeEnv(number_of_turns=3)

    cube.render()

    actions_to_leaf = None
    while not actions_to_leaf:
        actions_to_leaf = mcts.train(cube)

    action_space = cube.get_action_space()
    print([action_space[action_index] for action_index in actions_to_leaf])
