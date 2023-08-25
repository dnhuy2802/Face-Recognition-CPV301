### App Constants ###
import os

# Directories
CACHE_DIR = os.path.join(os.getcwd(), 'app', 'resources', 'data')
MODEL_TABLE_PATH = os.path.join(
    os.getcwd(), 'app', 'resources', 'model_table.csv')
MODEL_DIR = os.path.join(os.getcwd(), 'app', 'resources', 'model')

# Models
ML_MODEL_EXT = '.pkl'
DL_MODEL_EXT = '.h5'

# Image Constants
IMAGE_EXT = '.jpg'
TRAINING_IMAGE_SIZE = 112
