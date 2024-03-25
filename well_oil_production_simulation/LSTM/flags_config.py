# flags_config.py
from absl import flags

flags.DEFINE_string(
    "TARGET_WELL", default="default_value", help="Description of TARGET_WELL"
)
flags.DEFINE_integer("EPOCHS", default=10, help="Number of training epochs")
flags.DEFINE_integer("BATCH_SIZE", default=32, help="Batch size for training")
flags.DEFINE_integer("OBSERVATION_DATE", default=10, help="Observation date")
flags.DEFINE_float(
    "INFERENCE_GAUSSIAN_STD",
    default=0.1,
    help="Standard deviation for Gaussian noise during inference",
)
FLAGS = flags.FLAGS
