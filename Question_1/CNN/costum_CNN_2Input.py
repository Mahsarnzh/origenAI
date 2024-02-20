import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, concatenate

input_shape_cnn = (300, 7, 1)
input_shape_fc = (10, 15, 25, 24)
output_shape = (300, 7, 1)

# Input layers
input_cnn1 = Input(shape=input_shape_cnn, name='input_cnn1')
input_cnn2 = Input(shape=input_shape_fc, name='input_cnn2')

# CNN layers for both inputs
cnn_layer1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_cnn1)
cnn_layer2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_cnn2)

flatten_cnn1 = Flatten()(cnn_layer1)
flatten_cnn2 = Flatten()(cnn_layer2)

# Concatenating CNN outputs
concatenated = concatenate([flatten_cnn1, flatten_cnn2])

# Fully Connected layer after concatenation
final_fc_layer = Dense(2100, activation='linear')(concatenated)

# Reshape to match the desired output shape
reshaped_output = Reshape((300, 7))(final_fc_layer)

model = Model(inputs=[input_cnn1, input_cnn2], outputs=reshaped_output)

# Display the model summary
model.summary()

# Compile the model
custom_optimizer = Adam(learning_rate=0.0001)

# Compile the model with the custom optimizer
model.compile(optimizer=custom_optimizer, loss='mse', metrics=['accuracy'])

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
    X_train_cnn = input_data.copy()
    X_train_fc = input_data_BHP.copy()
    y = target_data.copy()

    X_train_cnn = np.array(X_train_cnn)
    X_train_fc = np.array(X_train_fc)
    y = np.array(y)
    return X_train_cnn, X_train_fc, y


X_train_cnn, X_train_fc, y = load_data()

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).reshape(y.shape)


X_train_fc_scaler = StandardScaler()
X_train_fc = X_train_fc_scaler.fit_transform(X_train_fc.reshape(-1, 1)).reshape(X_train_fc.shape)


num_samples, height, width, depth, channels = X_train_cnn.shape

# Reshape to apply StandardScaler along the last axis
X_train_cnn_reshaped = X_train_cnn.reshape((num_samples, -1, channels))

# Apply StandardScaler along the last axis
scaler = StandardScaler()
for i in range(channels):
    X_train_cnn_reshaped[:, :, i] = scaler.fit_transform(X_train_cnn_reshaped[:, :, i])

# Reshape back to the original shape
X_train_cnn_normalized = X_train_cnn_reshaped.reshape((num_samples, height, width, depth, channels))

# Apply the specified train-test-validation split
train_indices = range(0, 450)
test_indices = range(450, 475)
val_indices = range(475, 500)

X_train_cnn_split, X_test_cnn_split, X_val_cnn_split = X_train_cnn[train_indices], X_train_cnn[test_indices], X_train_cnn[val_indices]
X_train_fc_split, X_test_fc_split, X_val_fc_split = X_train_fc[train_indices], X_train_fc[test_indices], X_train_fc[val_indices]
y_train_split, y_test_split, y_val_split = y[train_indices], y[test_indices], y[val_indices]

# Train the model
model.fit(x={'input_cnn1': X_train_fc_split, 'input_cnn2': X_train_cnn_normalized[train_indices]}, y=y_scaled[train_indices],
          epochs=20, batch_size=32, validation_data=({'input_cnn1': X_val_fc_split, 'input_cnn2': X_train_cnn_normalized[val_indices]}, y_scaled[val_indices]))

# Train the model
# model.fit(x={'input_cnn1': X_train_fc, 'input_cnn2': X_train_cnn_normalized}, y=y_scaled, epochs=20, batch_size=32, validation_split=0.2)


model.save('costume_model_2inputs.h5')
