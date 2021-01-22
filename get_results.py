import time
from cubeEnv import CubeEnv
from mcts import MCTS
from model import buildModel, compile_model
from tensorflow import keras

if __name__ == "__main__":

    last_hope_model = keras.models.load_model("last_hope.hdf5")
    rnn_model = keras.models.load_model("rnn_model.hdf5")
    mcts_last_hope = MCTS(last_hope_model)
    mcts_rnn = MCTS(rnn_model)
    result_file  = open("results.txt", "w+") 
    max_n_turns = 8
    n_cubes = 20
    hour_in_seconds = 60 * 25
    result_file.write(f"numberOfTurns,avg_time_last_hope,percent_solved_last_hope,avg_time_rnn,percent_solved_rnn\n")
    for i in range(1, max_n_turns):
        times_last_hope = 0
        solved_last_hope = 0
        times_rnn = 0
        solved_rnn = 0
        for j in range(n_cubes):
            cube = CubeEnv(number_of_turns=i)

            start_time = time.time()
            naive_action_indexes = None
            while not naive_action_indexes and start_time + hour_in_seconds > time.time():
                naive_action_indexes = mcts_last_hope.train(cube)
            end_time = time.time()

            if naive_action_indexes:
                solved_last_hope += 1
                times_last_hope += end_time - start_time

            start_time = time.time()
            naive_action_indexes = None
            while not naive_action_indexes and start_time + hour_in_seconds > time.time():
                naive_action_indexes = mcts_rnn.train(cube)
            end_time = time.time()

            if naive_action_indexes:
                solved_rnn += 1
                times_rnn += end_time - start_time


        avg_time_rnn = times_rnn / solved_rnn
        avg_solved_rnn = solved_rnn / n_cubes

        avg_time_last_hope = times_last_hope / solved_last_hope
        avg_solved_last_hope = solved_last_hope / n_cubes

        result_file.write(f"{i},{avg_time_last_hope},{avg_solved_last_hope},{avg_time_rnn},{avg_solved_rnn}\n")
    result_file.close()
