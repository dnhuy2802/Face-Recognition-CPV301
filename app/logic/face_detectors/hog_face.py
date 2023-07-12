import cv2 as cv
import numpy as np
import dlib
from .constants import DetectionBox


class HogFaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.previous_box = DetectionBox()

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
        cordinates = self.detector(gray_frame, 1)

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
        cordinates = self.detector(gray_frame, 1)
        return cordinates

    def get_cordinate(self, cordinate: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x = cordinate.left()
        y = cordinate.top()
        w = cordinate.width()
        h = cordinate.height()
        return (x, y, w, h)
