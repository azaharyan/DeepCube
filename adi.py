import numpy as np
import tensorflow as tf
from tensorflow import keras
from copy  import deepcopy
import datetime
from model import buildModel, compile_model
from cubeAgent import CubeAgent
import time

model_path = F"/content/gdrive/My Drive/save_model" 

def adi(iterations=10000):
    model = buildModel(20*24)
    compile_model(model, 0.001)
    for it in range(iterations):

        # generate N scrambled cubes
        l = 100
        k = 20
        cube_agent = CubeAgent(number_of_cubes=l, batches=k)

        start_time = time.time()
        cube_agent.scramble_cubes_for_data()
        end_time = time.time()

        print(end_time-start_time)

        
        cubes = np.array(cube_agent.env).flatten()

        #initialize the training parameters -> marked by X and Y in the paper
        encodedStates = np.empty((k*l, 20*24)) 
        values = np.empty((k*l, 1))
        policies = np.empty(k*l)

        # iterate through the number of cubes and the number of actions
        i = 0
        for _ , state in enumerate(cubes):
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
            i += 1
        
        if it % 500 == 0:
          print("Iteration ", it)
        #log into Tensorboad
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        sample_weight = np.array([[1/(i+1) for i in range(k)] for j in range(l)]).flatten()

        model.fit(encodedStates,
                 { "output_policy": policies, "output_value": values },
                 epochs=15, sample_weight=sample_weight)
        model.save(model_path)
    return model

if __name__ == "__main__":

    model = adi()
    