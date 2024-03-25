from tensorflow.keras.layers import GaussianNoise
import torch.utils.data
from keras.callbacks import EarlyStopping
from absl import flags
from datetime import datetime
from os.path import join as pjoin
import torch
import torch.utils.data
from model_definition import get_model
import tensorflow as tf
from load_data import load_and_process_data
import numpy as np


FLAGS = flags.FLAGS

# Mark the flags as parsed to prevent duplicate definition error
flags.DEFINE_string(
    "TARGET_WELL", default="default_value", help="Description of TARGET_WELL"
)
flags.DEFINE_integer("EPOCHS", default=300, help="Number of training epochs")
flags.DEFINE_integer("BATCH_SIZE", default=32, help="Batch size for training")
flags.DEFINE_integer("OBSERVATION_DATE", default=10, help="Observation date")
flags.DEFINE_float(
    "INFERENCE_GAUSSIAN_STD",
    default=0.1,
    help="Standard deviation for Gaussian noise during inference",
)

FLAGS.mark_as_parsed()


df_t, target_df_t = load_and_process_data()


def _get_log_path():
    return pjoin("logs", f"well_{FLAGS.TARGET_WELL}")


def _get_result_path():
    return pjoin("result", f"well_{FLAGS.TARGET_WELL}")


def train_model():

    train_indices = range(0, 450)
    test_indices = range(450, 475)
    val_indices = range(475, 500)

    df_input = df_t
    df_targets = target_df_t
    # processor.remove_zero_wopr(well)

    # Convert DataFrames to PyTorch tensors
    input_data = torch.tensor(df_input.values, dtype=torch.float32)
    targets = torch.tensor(df_targets.values, dtype=torch.float32)

    # Split the data into training, testing, and validation sets
    input_train, targets_train = input_data[train_indices], targets[train_indices]
    input_test, targets_test = input_data[test_indices], targets[test_indices]
    input_val, targets_val = input_data[val_indices], targets[val_indices]

    # Convert NumPy arrays to PyTorch tensors
    train_x = torch.tensor(input_train, dtype=torch.float32)
    val_x = torch.tensor(input_val, dtype=torch.float32)
    test_x = torch.tensor(input_test, dtype=torch.float32)

    train_y = torch.tensor(targets_train, dtype=torch.float32)
    val_y = torch.tensor(targets_val, dtype=torch.float32)
    test_y = torch.tensor(targets_test, dtype=torch.float32)

    print("Shapes before calling get_model:")
    print("train_x shape:", train_x.shape)
    print("train_y shape:", train_y.shape)

    train_data = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=FLAGS.BATCH_SIZE, shuffle=True
    )

    val_data = torch.utils.data.TensorDataset(val_x, val_y)

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=FLAGS.BATCH_SIZE, shuffle=False
    )

    # Convert PyTorch tensors to NumPy arrays
    train_x_numpy = train_x.numpy()
    train_y_numpy = train_y.numpy()
    val_x_numpy = val_x.numpy()
    val_y_numpy = val_y.numpy()

    lstm_model = get_model(
        params={
            "input_shape": (
                train_x_numpy.shape[1],
                1,
            ),  # Assuming each feature corresponds to a time step
            "lstm1_units": 50,
            "lstm2_units": 50,
            "gaussian_std": 0.1,
            "dropout_rate": 0.2,
            "optimizer": "adam",
            "loss": "mean_squared_error",
        }
    )

    latest = tf.train.latest_checkpoint(_get_log_path())
    print(f"latest checkpoint: {latest}")

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=6, verbose=1
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = pjoin(_get_log_path(), "scalars")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    checkpoint_path = pjoin(_get_log_path(), "model.ckpt-{epoch:04d}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1
    )

    print("Training LSTM model using PyTorch DataLoader")
    for epoch in range(FLAGS.EPOCHS):
        for batch_x, batch_y in train_loader:
            # Convert PyTorch tensors to NumPy arrays for each batch
            batch_x_numpy = batch_x.numpy()
            batch_y_numpy = batch_y.numpy()
            print(f"epoch:{epoch}")
            # Train the model on the current batch
            history = lstm_model.train_on_batch(batch_x_numpy, batch_y_numpy)
            # print(f'history:{history}')

        print("Training has been ended")

        lstm_model.save_weights(checkpoint_path.format(epoch=FLAGS.EPOCHS))
        print("Model weights are saved")
        val_x_list = []
        val_y_list = []

        # Iterate over the DataLoader and convert to TensorFlow tensors
        for batch in val_loader:
            batch_x, batch_y = batch
            # Convert PyTorch tensors to NumPy arrays and then to TensorFlow tensors
            val_x_list.append(tf.convert_to_tensor(batch_x.numpy(), dtype=tf.float32))
            val_y_list.append(tf.convert_to_tensor(batch_y.numpy(), dtype=tf.float32))

        # Concatenate the lists to get complete tensors
        val_x = tf.concat(val_x_list, axis=0)
        val_y = tf.concat(val_y_list, axis=0)

        early_stopping = EarlyStopping(
            monitor="loss",  # Change 'val_loss' to 'loss'
            restore_best_weights=True,
        )

        # Assuming val_loader is defined, use it for validation
        history = lstm_model.fit(
            val_x,
            val_y,
            epochs=40,  # You may adjust the number of epochs for validation
            use_multiprocessing=True,
            workers=8,
            callbacks=[es, tensorboard_callback, early_stopping],
        )

        print("Validation has been ended")

    def cascade_inference(model, test_x, test_y, obs, gaussian_std):
        y_hat_list = []
        observed = test_y[:obs]
        y_hat_list.extend(observed)

        buffer = test_x[obs : obs + 1]
        mu = gaussian_std

        for i in range(obs + 1, 12):
            # print(i)
            # buffer = buffer.reshape(-1, 1)
            y_hat = lstm_model(buffer).numpy()
            predicted_val = y_hat[0, 0]
            y_hat_list.append(predicted_val)

            buffer = np.delete(buffer, 0, 1)
            next_wbhp = test_x[i : i + 1][0, 4]
            predicted_array = np.array([[predicted_val, next_wbhp]])

            buffer = buffer.reshape(-1, 1)
            # Concatenate along axis 0
            buffer = np.vstack((buffer, predicted_array.T))
            # buffer = np.reshape(buffer, (1, 5, 2))

            mean_wopr = np.mean(buffer, axis=0)
            mean_wbhp = np.mean(buffer, axis=1)

            # Assuming buffer[0] and buffer[1] are 1D arrays representing sequences
            wopr_predicted_noise_added = np.random.normal(mean_wopr, mu, len(buffer[0]))
            wbhp_predicted_noise_added = np.random.normal(
                np.mean(mean_wbhp), mu, len(buffer[1])
            )
            buffer[0] = wopr_predicted_noise_added
            buffer[1] = wbhp_predicted_noise_added

        return y_hat_list

    ## Inference
    y_hat_list = cascade_inference(
        lstm_model,
        test_x.numpy(),
        test_y.numpy(),
        obs=FLAGS.OBSERVATION_DATE,
        gaussian_std=FLAGS.INFERENCE_GAUSSIAN_STD,
    )
    print("Saving inference results...")
