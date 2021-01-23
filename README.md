# DeepCube
Implementation of DeepCube paper that solves the Rubik's Cube using Deep Reinforcement Learning

## Running
To run the solver all you must do is to run the main.py file. It will created scrambled cube and will try to solve it. The created cube will scramble number_of_turns(a parameter of the environment) times.
The function keras.models.load_model has parameter a name of a neural network name.
You can create a model instead of using saved one by uncommenting the commented part of the main.py file and removing the line for loading a model.

To train a neural network executing adi.py is enough. However one my want to tweak the hyperparameters there(iterations,l,k).

## Prerequisites
1. tensorflow
2. pycuber
3. numpy
4. matplotlib
