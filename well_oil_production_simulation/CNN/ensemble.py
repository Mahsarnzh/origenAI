import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import sys
from tensorflow.keras import layers, models
from sklearn.base import clone
from tensorflow.keras.layers import Reshape


# Define the base 1D CNN model
def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(
        layers.Conv3D(
            32, (3, 3, 3), activation="relu", input_shape=input_shape, padding="same"
        )
    )
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Conv3D(64, (3, 3, 3), activation="tanh", padding="same"))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Conv3D(128, (3, 3, 3), activation="relu", padding="same"))
    model.add(layers.Conv3D(128, (3, 3, 3), activation="tanh", padding="same"))
    model.add(layers.Conv3D(128, (3, 3, 3), activation="tanh", padding="same"))
    model.add(layers.Conv3D(128, (3, 3, 3), activation="relu", padding="same"))
    model.add(layers.Flatten())
    model.add(layers.Dense(300 * 7, activation="linear"))
    model.add(layers.Reshape((300, 7)))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae", "accuracy"])
    return model


def create_lstm_model(X, y):
    X = X.reshape((X.shape[0], X.shape[1], -1))  # (500, 10, 15 * 25 * 24)
    model = Sequential()
    model.add(LSTM(units=128, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(units=300 * 7, activation="linear"))
    model.add(Reshape((300, 7)))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy'])
    return model


# noinspection PyPep8Naming
def load_data():
    from data_loading import load_data as ld

    data, target_data, bhp_data = ld()
    X, y = [*data.values()], [*target_data.values()]
    target_data = np.array(
        [
            target_data[in_nm, sim_nm]
            for in_nm, sim_nm in sorted(
                [(in_nm, sim) for in_nm, sim in target_data.keys()],
                key=lambda s: int(s[1].split("_")[0][3:]),
            )  # then sort by simulation number
        ]
    )
    assert target_data.shape == (
        500,
        300,
        7,
    ), "malformed target data of shape %s" % str(target_data.shape)

    input_data_BHP = np.array(
        [
            bhp_data[in_nm, sim_nm]
            for in_nm, sim_nm in sorted(
                [(in_nm, sim) for in_nm, sim in bhp_data.keys()],
                key=lambda s: int(s[1].split("_")[0][3:]),
            )  # then sort by simulation number
        ]
    )
    assert input_data_BHP.shape == (
        500,
        300,
        7,
    ), "malformed target data of shape %s" % str(input_data_BHP.shape)

    num_inputs = 500

    in_nms = sorted(set(k[0] for k in data.keys()))
    input_data = np.array(
        [
            [data[in_nm, "sim%d_%s" % (i + 1, in_nm.lower())] for in_nm in in_nms]
            for i in range(num_inputs)
        ]
    )

    assert input_data.shape == (
        500,
        10,
        15,
        25,
        24,
    ), "malformed input data of shape %s" % str(input_data.shape)

    # Append values to X and y lists
    X = input_data.copy()
    y = target_data.copy()

    X = np.array(X)
    y = np.array(y)
    print("Number of samples in X:", X.shape[0])
    print("Number of samples in y:", y.shape[0])

    return X, y


# noinspection PyPep8Naming,PyUnresolvedReferences
def main():
    input_shape = (10, 15, 25, 24)
    X_, y_ = load_data()

    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y_.reshape(-1, 1)).reshape(y_.shape)
    num_samples, height, width, depth, channels = X_.shape
    # Reshape to apply StandardScaler along the last axis
    X = X_.reshape((num_samples, -1, channels))

    # Apply StandardScaler along the last axis
    scaler = StandardScaler()
    for i in range(channels):
        X[:, :, i] = scaler.fit_transform(X[:, :, i])

    # Reshape back to the original shape
    X = X.reshape((num_samples, height, width, depth, channels))

    # Adjust indices for training-test-validation split
    train_indices = range(1, 451)
    test_indices = range(451, 476)
    val_indices = range(476, 501)

    # Create LSTM model
    num_lstm_models = 2  # Adjust as needed
    lstm_models = [create_lstm_model(X, y) for _ in range(num_lstm_models)]

    for i, model in enumerate(lstm_models):
        print(f"Training LSTM Model {i + 1}...")
        X_train, y_train = X[train_indices], y[train_indices]
        # Reshape input data for LSTM
        X_train_reshaped = X_train.reshape((X_train.shape[0], 10, 9000))
        model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32)

    num_models = 3
    cnn_models = [create_cnn_model(input_shape=input_shape) for _ in range(num_models)]

    # Train each CNN model on different subsets of the data
    for i, model in enumerate(cnn_models):
        print(f"Training CNN Model {i + 1}...")
        X_train, y_train = X[train_indices], y[train_indices]
        model.fit(X_train, y_train, epochs=2, batch_size=32)

    # Create an ensemble of traditional ML models
    base_models = [
        # ("rf", RandomForestRegressor(n_estimators=50, random_state=42)),
        # ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
        (
            "mlp",
            MLPRegressor(
                hidden_layer_sizes=(50,),
                max_iter=500,
                random_state=42,
                solver="adam",
                tol=1e-4,
            ),
        ),
    ]

    # Train each traditional ML model on different subsets of the data
    ml_models = [clone(model) for _, model in base_models]
    for i, model in enumerate(ml_models):
        print(f"Training Model {i + 1}...")
        # indices = np.random.choice(
        #     range(X.shape[0]), size=int(X.shape[0] * 0.8), replace=False
        # )
        X_train, y_train = X[train_indices], y[train_indices]
        if len(X_train.shape) > 2:
            # Flatten input data for MLP
            X_train_flattened = X_train.reshape((X_train.shape[0], -1))
        else:
            X_train_flattened = X_train
        # noinspection PyUnresolvedReferences
        model.fit(X_train_flattened, y_train.reshape((y_train.shape[0], -1)))

    # Create a StackingRegressor with a Linear Regression meta-model
    stacked_model = StackingRegressor(
        estimators=base_models, final_estimator=LinearRegression()
    )

    # Predictions from LSTM models
    lstm_predictions = np.concatenate(
        [model.predict(X.reshape((X.shape[0], 10, 9000))) for model in lstm_models],
        axis=1,
    )
    # Predictions from CNN models
    cnn_predictions = np.concatenate([model.predict(X) for model in cnn_models], axis=1)
    # Predictions from MLP models
    mlp_predictions = np.concatenate(
        [model.predict(X.reshape((X.shape[0], -1))) for model in ml_models], axis=1
    )

    # Combine all predictions for input to the StackingRegressor
    stacked_inputs = np.concatenate(
        [lstm_predictions, cnn_predictions, mlp_predictions.reshape(500, 300, 7)],
        axis=1,
    )
    y_reshaped = y.reshape(-1, 1)
    shape_y = y_reshaped.shape[0]
    num_modles = len(
        [lstm_predictions, cnn_predictions, mlp_predictions.reshape(500, 300, 7)]
    )
    reshaped_stacked_inputs = stacked_inputs.reshape(shape_y, num_modles)

    # Train the StackingRegressor on the combined predictions
    stacked_model.fit(reshaped_stacked_inputs, y_reshaped)

    # Make predictions with the StackingRegressor
    stacked_predictions = stacked_model.predict(reshaped_stacked_inputs)
    mse = mean_squared_error(y_reshaped, stacked_predictions)
    print(f"Mean Squared Error: {mse}")

    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error: {rmse}")


if __name__ == "__main__":
    sys.exit(main())
