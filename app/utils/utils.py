### Utils ###

import uuid
import string
import base64
from unidecode import unidecode
import numpy as np
import cv2 as cv


def debug_print(*args, **kwargs):
    print(*args, **kwargs)


def generate_uuid(length=5):
    return str(uuid.uuid4())[:length]


def get_idetifier(text: str):
    return unidecode(text.replace(" ", "_").lower())


def convert_base64_to_image(base64_string):
    try:
        # Convert base64 to np.ndarray
        encoded_data = base64_string.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), dtype=np.uint8)
        image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        return image
    except Exception as e:
        debug_print(e)
        return None


def convert_to_gray(image: np.ndarray):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
