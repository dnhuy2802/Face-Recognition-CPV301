import cv2 as cv
import numpy as np
from .constants import HAAR_CASCADE_PATH, DetectionBox


class HaarCascadeDetector:
    def __init__(self, face_cascade_path: str = HAAR_CASCADE_PATH):
        self.detector = cv.CascadeClassifier(face_cascade_path)
        self.previous_box = DetectionBox()
        self.scale_factor = 1.3
        self.min_neighbors = 3

    def detect_face(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        # Convert to grayscale if needed
        if len(frame.shape) > 2:
            gray_frame: np.ndarray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray_frame: np.ndarray = frame

        # Get initial box cordinates
        if len(self.previous_box.cordinates) == 0:
            self.previous_box.get_box_default_points(gray_frame.shape)

        # Get face cordinates from frame
        cordinates = self.detector.detectMultiScale(
            gray_frame, self.scale_factor, self.min_neighbors)

        # Return previous box if no face is detected
        if len(cordinates) > 0:
            self.previous_box.set_cordinates(self.get_cordinate(cordinates[0]))
        return self.previous_box.cordinates

    def detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        # Convert to grayscale if needed
        if len(frame.shape) > 2:
            gray_frame: np.ndarray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray_frame: np.ndarray = frame

        # Get face cordinates from frame
        cordinates = self.face_cascade.detectMultiScale(
            gray_frame, self.scale_factor, self.min_neighbors)
        return cordinates

    def get_cordinate(self, cordinate: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x = cordinate[0]
        y = cordinate[1]
        w = cordinate[2]
        h = cordinate[3]
        return (x, y, w, h)
