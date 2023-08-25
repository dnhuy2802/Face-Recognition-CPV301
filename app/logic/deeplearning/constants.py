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
INPUT_SHAPE = (TRAINING_IMAGE_SIZE * 2, TRAINING_IMAGE_SIZE * 2, 3)
MOMENTUM = 0.9
NUM_CLASSES = int()
PEOPLE_LABELS = list()

# Set the paths to your dataset
TEMP_DATA_DIR = os.path.join(os.getcwd(), "app", "temp")
TRAIN_DATA_DIR = os.path.join(os.getcwd(), "app", "temp", "train")
VALIDATION_DATA_DIR = os.path.join(os.getcwd(), "app", "temp", "val")
TEST_DATA_DIR = os.path.join(os.getcwd(), "app", "temp", "test")
RESOURCES_DIR = os.path.join(os.getcwd(), "app", "resources", "data")

# Text Variables
RED = (0, 0, 255)
GREEN = (0, 255, 0)
FONT_SCALE = 0.5

# Face Detection Variables
CONF_THRESHOLD = 0.8
PERCENT_EXPAND = 0.0
CONF_FACE_THRESHOLD = 0.9
