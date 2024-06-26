import tensorflow as tf
from tensorflow.keras.layers import GaussianNoise, LSTM, Dropout, Dense
from tensorflow.keras.layers import Flatten


def get_model(params):
    input_shape = params["input_shape"]
    lstm1_units = params["lstm1_units"]
    lstm2_units = params["lstm2_units"]
    gaussian_std = params["gaussian_std"]
    dropout_rate = params["dropout_rate"]
    optimizer = params["optimizer"]
    loss = params["loss"]

    model = tf.keras.models.Sequential()
    model.add(GaussianNoise(gaussian_std, input_shape=input_shape))

    model.add(LSTM(lstm1_units, activation="relu", return_sequences=True))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(lstm2_units, activation="relu", return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    # model.add(Dense(2100, activation='linear'))
    model.add(Dense(600, activation="linear"))

    model.compile(optimizer=optimizer, loss=loss)

    model.summary()

    return model
