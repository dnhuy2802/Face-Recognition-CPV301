### Local Constants ###
import os
from ...utils.constants import TRAINING_IMAGE_SIZE

IMAGE_SIZE = (TRAINING_IMAGE_SIZE, TRAINING_IMAGE_SIZE)
LOCAL_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
