# import sys
#
# import numpy as np
# from tensorflow.keras import layers, models
# from data_loading import load_data
#
#
# # noinspection PyPep8Naming
# def main():
#
#     data, target_data, BHP_data = load_data()
#
#     X = []
#     y = []
#
#     for key, value in data.items():
#         X.append(value)  # Flatten the input data
#
#     for key, value in target_data.items():
#         y.append(value)
#
#
#     target_data = np.array(
#         [
#             target_data[in_nm, sim_nm]
#             for in_nm, sim_nm in sorted(
#                 [
#                     (in_nm, sim)
#                     for in_nm, sim in target_data.keys()
#                     # if nm == "WOPR"
#                 ],
#                 key=lambda s: int(s[1].split("_")[0][3:]),
#             )  # then sort by simulation number
#         ]
#     )
#     assert target_data.shape == (
#         500,
#         300,
#         7,
#     ), "malformed target data of shape %s" % str(target_data.shape)
#
#
#
#
#     input_data = np.array(
#         [
#             BHP_data[in_nm, sim_nm]
#             for in_nm, sim_nm in sorted(
#                 [
#                     (in_nm, sim)
#                     for in_nm, sim in BHP_data.keys()
#                     # if nm == "WOPR"
#                 ],
#                 key=lambda s: int(s[1].split("_")[0][3:]),
#             )  # then sort by simulation number
#         ]
#     )
#     assert input_data.shape == (
#         500,
#         300,
#         7,
#     ), "malformed target data of shape %s" % str(input_data.shape)
#
#
#
#     # # Create 500 samples of input variables with shape (10, 15, 25, 24)
#     # num_inputs = 500
#     # input_shape = (10, 15, 25, 24)
#     #
#     # in_nms = sorted(set(k[0] for k in data.keys()))
#     # input_data = np.array(
#     #     [
#     #         [
#     #             data[in_nm, "sim%d_%s" % (i+1, in_nm.lower())]
#     #             for in_nm in in_nms
#     #         ]
#     #         for i in range(num_inputs)
#     #     ]  # sort the input by friendly name
#     # )
#     #
#     # assert input_data.shape == (
#     #     500,
#     #     10,
#     #     15,
#     #     25,
#     #     24,
#     # ), "malformed input data of shape %s" % str(input_data.shape)
#
#     # Append values to X and y lists
#     X = input_data.copy()  # [value for key, value in input_data.items()]
#     y = target_data.copy()
#
#     # Assuming you have loaded your data into X (input) and y (target)
#     # X should have shape (500, 10, 15, 25, 24)
#     # y should have shape (500, 300, 7)
#     X = np.array(X)
#     y = np.array(y)
#     print("Number of samples in X:", X.shape[0])
#     print("Number of samples in y:", y.shape[0])
#     input_shape = (500, 300, 7)
#
#     # Define the 3D CNN model
#     input_shape = (300, 7)
#
#     model = models.Sequential()
#     model.add(layers.RepeatVector(500, input_shape=input_shape))  # Repeat along a new axis
#     model.add(layers.Conv1D(32, 3, activation="relu", padding="same"))
#     model.add(layers.MaxPooling1D(2))
#     model.add(layers.Conv1D(64, 3, activation="tanh", padding="same"))
#     model.add(layers.MaxPooling1D(2))
#     model.add(layers.Conv1D(128, 3, activation="relu", padding="same"))
#     model.add(layers.Conv1D(128, 3, activation="relu", padding="same"))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(300 * 7, activation="linear"))
#     model.add(layers.Reshape((300, 7)))
#
#     # Compile the model
#     model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
#
#     # Specify the indices for training, testing, and validation
#     train_indices = range(0, 450)
#     test_indices = range(450, 475)
#     val_indices = range(475, 500)
#
#     # Split the data into training, testing, and validation sets
#     X_train, y_train = X[train_indices], y[train_indices]
#     X_test, y_test = X[test_indices], y[test_indices]
#     X_val, y_val = X[val_indices], y[val_indices]
#
#     # Train the model
#     model.fit(
#         X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val)
#     )
#
#
# if __name__ == "__main__":
#     sys.exit(main())



import sys
import numpy as np
from tensorflow.keras import layers, models
from data_loading import load_data

# noinspection PyPep8Naming
def main():
    data, target_data, BHP_data = load_data()

    X = []
    y = []

    for key, value in data.items():
        X.append(value)  # Flatten the input data

    for key, value in target_data.items():
        y.append(value)

    target_data = np.array([
        target_data[in_nm, sim_nm]
        for in_nm, sim_nm in sorted(
            [
                (in_nm, sim)
                for in_nm, sim in target_data.keys()
            ],
            key=lambda s: int(s[1].split("_")[0][3:]),
        )  # then sort by simulation number
    ])
    assert target_data.shape == (
        500,
        300,
        7,
    ), "malformed target data of shape %s" % str(target_data.shape)

    input_data = np.array([
        BHP_data[in_nm, sim_nm]
        for in_nm, sim_nm in sorted(
            [
                (in_nm, sim)
                for in_nm, sim in BHP_data.keys()
            ],
            key=lambda s: int(s[1].split("_")[0][3:]),
        )  # then sort by simulation number
    ])
    assert input_data.shape == (
        500,
        300,
        7,
    ), "malformed target data of shape %s" % str(input_data.shape)

    # Append values to X and y lists
    X = input_data.copy()
    y = target_data.copy()

    # Assuming you have loaded your data into X (input) and y (target)
    # X should have shape (500, 300, 7)
    # y should have shape (500, 300, 7)
    X = np.array(X)
    y = np.array(y)
    print("Number of samples in X:", X.shape[0])
    print("Number of samples in y:", y.shape[0])

    # Define the 1D CNN model
    input_shape = (300, 7)

    model = models.Sequential()
    model.add(layers.Conv1D(32, 3, activation="relu", padding="same", input_shape=input_shape))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(64, 3, activation="tanh", padding="same"))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(128, 3, activation="relu", padding="same"))
    model.add(layers.Conv1D(128, 3, activation="relu", padding="same"))
    model.add(layers.Flatten())
    model.add(layers.Dense(300 * 7, activation="linear"))
    model.add(layers.Reshape((300, 7)))

    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

    # Specify the indices for training, testing, and validation
    train_indices = range(0, 450)
    test_indices = range(450, 475)
    val_indices = range(475, 500)

    # Split the data into training, testing, and validation sets
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

if __name__ == "__main__":
    sys.exit(main())
