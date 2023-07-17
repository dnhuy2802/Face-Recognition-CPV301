import os
from ...utils.constants import TRAINING_IMAGE_SIZE

# Specify the batch size and image dimensions
DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VALID_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.2
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
DEFAULT_FINE_TUNE = 0
LEARNING_RATE = 0.001
DECAY = LEARNING_RATE / DEFAULT_EPOCHS
IMAGE_SIZE = tuple([TRAINING_IMAGE_SIZE, TRAINING_IMAGE_SIZE])
INPUT_SHAPE = tuple([TRAINING_IMAGE_SIZE, TRAINING_IMAGE_SIZE, 3])
MOMENTUM = 0.9
NUM_CLASSES = int()
PEOPLE_LABELS = list()

# Set the paths to your dataset
LOGIC_DIR = os.path.join(os.getcwd(), "app", "logic", "deeplearning")
TRAIN_DATA_DIR = os.path.join(LOGIC_DIR, "temp", "train")
VALID_DATA_DIR = os.path.join(LOGIC_DIR, "temp", "valid")
TEST_DATA_DIR = os.path.join(LOGIC_DIR, "temp", "test")
TEMP_DATA_DIR = os.path.join(os.getcwd(), "app", "temp")
ASSETS_DIR = os.path.join(LOGIC_DIR, "assets")

# Text Variables
RED = (0, 0, 255)
GREEN = (0, 255, 0)
FONT_SCALE = 0.5

# Face Detection Variables
CONF_THRESHOLD = 0.8
PERCENT_EXPAND = 0.0
CONF_FACE_THRESHOLD = 0.9
