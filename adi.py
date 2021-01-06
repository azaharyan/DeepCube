import numpy as np

from cubeAgent import CubeAgent
from model import buildModel, compile_model


def adi(iterations=100):
    model = buildModel(20 * 24)
    compile_model(model, 0.001)

    for _ in range(iterations):

        cube_agent = CubeAgent(number_of_cubes=150)
        # generate number_of_cubes * incrementally scrambled cubes (up to batches)
        cube_agent.scramble_cubes_for_data()

        for cubes in cube_agent.env:

            # initialize the training parameters -> marked by X and Y in the paper
            encoded_states = np.empty((len(cubes), 20 * 24))
            values = np.empty((len(cubes), 1))
            policies = np.empty(len(cubes))

            # iterate through the number of cubes and the number of actions
            for i, state in enumerate(cubes):
                values_for_state = np.zeros(len(state.action_space))

                encoded_states[i] = np.array(state.get_one_hot_state().flatten())
                actions = state.action_space

                start_state = state.cube.copy()

                # 1-depth BFS
                for j, action in enumerate(actions):
                    _, reward = state.step(j)
                    child_state_encoded = np.array(state.get_one_hot_state()).flatten()
                    state.set_state(start_state)  # set back to the original

                    value, _ = model.predict(child_state_encoded[None, :])
                    value_number = value[0][0]
                    values_for_state[j] = value_number + reward

                values[i] = np.array([values_for_state.max()])
                policies[i] = values_for_state.argmax()

            # log into Tensorboad
            # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            model.fit(encoded_states, {"output_policy": policies, "output_value": values}, epochs=15)
            model.save('saved_model')
    return model


if __name__ == "__main__":
    model = adi()
