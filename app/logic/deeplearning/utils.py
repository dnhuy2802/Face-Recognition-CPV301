import os
import cv2 as cv
import numpy as np
from ...utils.utils import generate_uuid
from ...utils.constants import IMAGE_EXT, CACHE_DIR


def save_to_temp(img: np.ndarray, face: str, folder_path: str):
    file_name = generate_uuid(10)
    # Generate face folder
    face_folder = os.path.join(folder_path, face)
    if not os.path.exists(face_folder):
        os.mkdir(face_folder)
    # Save image
    cv.imwrite(os.path.join(face_folder, file_name + IMAGE_EXT), img)


def get_labels():
    return [f for f in os.listdir(
            CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, f))]
