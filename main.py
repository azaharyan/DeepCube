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
    cube = CubeEnv(number_of_turns=1)
    result = None

    cube.render()

    while not result:
        result, actions = mcts.train(cube)

    result[0].render()
    print(actions[0].render())
