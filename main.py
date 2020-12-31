from cubeEnv import CubeEnv
from mcts import MCTS
from model import buildModel, compile_model

if __name__ == "__main__":
    model = buildModel(20 * 24)
    compile_model(model, 0.01)
    mcts = MCTS(model)
    cube = CubeEnv(number_of_turns=1)
    print(mcts.train(cube))
