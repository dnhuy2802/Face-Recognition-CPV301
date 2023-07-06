### Get Path of Images by Name ###

import os
from .constants import CACHE_DIR


def get_path(name: str):
    # Return the path of the person if it exists in the cache directory
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path):
        return path

    # Fetch images from the database if it does not exist in the cache directory
    # TODO: Fetch images from the database
