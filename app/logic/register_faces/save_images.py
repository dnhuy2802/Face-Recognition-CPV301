import os
import base64
from ...utils.constants import CACHE_DIR, IMAGE_EXT


def save_images(name: str, images: list[str]):
    # Create a folder with the name of the person
    path = os.path.join(CACHE_DIR, name)
    if not os.path.exists(path):
        os.mkdir(path)
    # Save the images to the folder
    image_idx = 0
    for image in images:
        image_bytes = base64.decodebytes(image.split(',')[1].encode('utf-8'))
        with open(os.path.join(path, f'{image_idx}{IMAGE_EXT}'), 'wb') as file:
            file.write(image_bytes)
        image_idx += 1
    # Return the path to the folder
    return path
