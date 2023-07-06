import os
import base64
import cv2 as cv
import numpy as np
from .constants import CACHE_DIR, IMAGE_SIZE


def save_images(name: str, images: list[str]):
    # Create a folder with the name of the person
    path = os.path.join(CACHE_DIR, name)
    if not os.path.exists(path):
        os.mkdir(path)
    # Save the images to the folder
    image_idx = 0
    for image in images:
        image_bytes = base64.decodebytes(image.split(',')[1].encode('utf-8'))
        with open(os.path.join(path, f'{image_idx}.jpg'), 'wb') as file:
            file.write(image_bytes)
        image_idx += 1
    # Return the path to the folder
    return path


def resize_images(path):
    # Get the images from the folder
    images = os.listdir(path)
    # Resize the images
    for image in images:
        # Load the image
        image_path = os.path.join(path, image)
        try:
            img: np.ndarray = cv.imread(image_path)
            # Crop the image
            height, width, _ = img.shape
            offset = np.abs(height - width) // 2
            img = img[:, offset:offset + height]
            img = cv.resize(img, IMAGE_SIZE)
            cv.imwrite(image_path, img)
        except Exception as e:
            print(e)
