# DeepCube
Tensorflow/Keras implementation of [DeepCube paper](https://arxiv.org/pdf/1805.07470.pdf) by McAleer et al. that solves the Rubik's Cube using Deep Reinforcement Learning. We used a feed-forward network and a RNN network to compare the result. Both models are augmented with Monte Carlo Tree Search for solving the cube itself.

## Prerequisites
* [Tensorflow - v2.4.0](https://github.com/tensorflow/tensorflow)
* [pycuber](https://github.com/adrianliaw/PyCuber)
* [numpy](https://github.com/numpy/numpy)
* [matplotlib](https://github.com/matplotlib/matplotlib)

## Running
To run the solver all you must do is to run the main.py file. It will created scrambled cube and will try to solve it. The created cube will scramble number_of_turns(a parameter of the environment) times.
The function keras.models.load_model has parameter a name of a neural network name.
You can create a model instead of using saved one by uncommenting the commented part of the main.py file and removing the line for loading a model.

To train a neural network executing adi.py is enough. However one my want to tweak the hyperparameters there(iterations,l,k).

