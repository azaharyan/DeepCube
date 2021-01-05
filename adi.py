import numpy as np
import tensorflow as tf
from tensorflow import keras
from copy  import deepcopy
from cubeAgent import CubeAgent
from model import buildModel, compile_model
import datetime

def adi(iterations=100):
    model = buildModel(20*24)
    compile_model(model, 0.001)
    for _ in range(iterations):

        # generate N scrambled cubes
        cubes = CubeAgent(number_of_cubes=150)
        cubes.scrabmle_cubes_for_data(number_of_turns=3)

        #initialize the training parameters -> marked by X and Y in the paper
        encodedStates = np.empty((len(cubes.env), 20*24)) 
        values = np.empty((len(cubes.env), 1))
        policies = np.empty(len(cubes.env))

        # iterate through the number of cubes and the number of actions
        for i, state in enumerate(cubes.env):
            valuesForState = np.zeros(len(state.action_space))

            encodedStates[i] = np.array(state.get_one_hot_state().flatten())
            actions = state.action_space

            start_state = state.cube.copy()

            #1-depth BFS 
            for j, action in enumerate(actions):
                _ , reward = state.step(j)
                childStateEncoded = np.array(state.get_one_hot_state()).flatten()
                state.set_state(start_state) #set back to the original

                value, _ = model.predict(childStateEncoded[None, :])
                valueNumber = value[0][0]
                valuesForState[j] = valueNumber + reward

            values[i] = np.array([valuesForState.max()])
            policies[i] = valuesForState.argmax()
        
        #log into Tensorboad
        #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model.fit(encodedStates, { "output_policy": policies, "output_value": values },
                 epochs=15)
        model.save('saved_model')
    return model

if __name__ == "__main__":

    model = adi()
    