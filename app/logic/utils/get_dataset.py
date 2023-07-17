import os
import cv2 as cv
import numpy as np
from ...utils.get_path import get_images_abspath
from ..utils.split_image import split_img
from ..face_detectors.hog_face import HogFaceDetector
from ...utils.constants import CACHE_DIR


### Get dataset in the form of (faces, labels) ###
def get_dataset(registered_faces: list[str], path_only: bool = False) -> tuple[np.ndarray[cv.Mat | str], np.ndarray[str]]:
    # Init face detector
    face_detector = HogFaceDetector()
    # Init empty array for faces and labels
    faces = []
    labels = []
    # Get dataset
    for face in registered_faces:
        face_paths = get_images_abspath(face)
        for face_path in face_paths:
            if path_only:
                faces.append(face_path)
                labels.append(face)
                continue
            img: cv.Mat = cv.imread(face_path, cv.IMREAD_GRAYSCALE)
            face_cordinate = face_detector.detect_face(img)
            face_image = split_img(img, face_cordinate)
            faces.append(face_image)
            labels.append(face)
    return np.array(faces), np.array(labels)


def get_all_faces_names() -> list[str]:
    return os.listdir(CACHE_DIR)
