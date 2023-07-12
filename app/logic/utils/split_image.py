import cv2 as cv
import numpy as np
from ...utils.constants import TRAINING_IMAGE_SIZE


def force_resize(img: np.ndarray) -> np.ndarray:
    return cv.resize(img, (TRAINING_IMAGE_SIZE, TRAINING_IMAGE_SIZE), interpolation=cv.INTER_AREA)


def split_img(img: np.ndarray, cordinate: tuple[int, int, int, int]) -> np.ndarray:
    (x, y, w, h) = cordinate
    return force_resize(img[y:y+h, x:x+w])
