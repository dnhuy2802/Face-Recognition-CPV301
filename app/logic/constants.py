import os

# Specify the batch size and image dimensions
BATCH_SIZE = 32
EPOCHS = 50
LRATE = 0.001
DECAY = LRATE / EPOCHS
IMG_HEIGHT, IMG_WIDTH = 224, 224
IMAGE_SIZE = tuple([IMG_HEIGHT, IMG_WIDTH])
MOMENTUM = 0.9
NUM_CLASSES = int()
PEOPLE_LABELS = list()

# Set the paths to your dataset
TRAIN_DATA_DIR = os.path.join(os.getcwd(), "app", "data", "train")
VALIDATION_DATA_DIR = os.path.join(os.getcwd(), "app", "data", "val")
TEST_DATA_DIR = os.path.join(os.getcwd(), "app", "data", "test")
TEMP_DATA_DIR = os.path.join(os.getcwd(), "app", "temp")
MODEL_DIR = os.path.join(os.getcwd(), "app", "model")
RESOURCES_DIR = os.path.join(os.getcwd(), "app", "resources")
DATA_DIR = os.path.join(os.getcwd(), "app", "data")

# Text Variables
RED = (0, 0, 255)
GREEN = (0, 255, 0)
FONT_SCALE = 0.5