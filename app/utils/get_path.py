### Get Path of Images by Name ###

import os
from .constants import CACHE_DIR, IMAGE_EXT


def get_abspath(name: str) -> str:
    # Return the path of the person if it exists in the cache directory
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path):
        return path
    else:
        raise FileNotFoundError(
            f'Person {name} does not exist in the cache directory')

    # Fetch images from the database if it does not exist in the cache directory
    # TODO: Fetch images from the database


def get_images_abspath(name: str) -> list[str]:
    # Return the path of the person if it exists in the cache directory
    path = get_abspath(name)
    if os.path.exists(path):
        abs_paths = []
        image_names = os.listdir(path)
        # Filter out the non-image files
        for image_name in image_names:
            if image_name.endswith(IMAGE_EXT):
                abs_paths.append(os.path.join(path, image_name))
        return abs_paths

    # Fetch images from the database if it does not exist in the cache directory
