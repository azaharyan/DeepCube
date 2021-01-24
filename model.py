import tensorflow as tf
from tensorflow import keras

def buildModel_rnn(input_size):
  input = keras.Input(shape=(input_size,))
  expanded = tf.expand_dims(input, axis=-1)
  first_layer = tf.keras.layers.LSTM(100, input_shape=(1,input_size))(expanded)
  second_layer_value = tf.keras.layers.Dense(256, activation='elu')(first_layer)
  second_layer_policy = tf.keras.layers.Dense(256, activation='elu')(first_layer)
  value_output = tf.keras.layers.Dense(1, name="output_value")(second_layer_value)
  policy_output = tf.keras.layers.Dense(12, activation='softmax', name="output_policy")(second_layer_policy)

  model = tf.keras.Model(inputs=input, outputs=[value_output, policy_output])
  return model

def buildModel(input_size):
    input = keras.Input(shape=(input_size,))

    first_layer = tf.keras.layers.Dense(4096, activation='elu')(input)
    second_layer = tf.keras.layers.Dense(2048, activation='elu')(first_layer)
    third_layer_value = tf.keras.layers.Dense(512, activation='elu')(second_layer)
    third_layer_policy = tf.keras.layers.Dense(512, activation='elu')(second_layer)

    value_output = tf.keras.layers.Dense(1, name="output_value")(third_layer_value)
    policy_output = tf.keras.layers.Dense(12, activation='softmax',
                                          name="output_policy")(third_layer_policy)

    model = tf.keras.Model(inputs=input, outputs=[value_output, policy_output])
    return model


def compile_model(model, learning_rate):
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.5
    )

    opt = keras.optimizers.RMSprop(learning_rate)
    model.compile(loss={'output_value': 'mean_squared_error', 'output_policy': 'sparse_categorical_crossentropy'},
                  optimizer=opt)
    model.summary()
