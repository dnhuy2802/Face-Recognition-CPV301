"""
- Capture 100 image face from camera using OpenCV
"""
import cv2
import os
import time
import numpy as np
from PIL import Image
from IPython.display import display


def capture_face(name: str):
    """
    Capture 100 image face from camera using OpenCV
    :param name: name of person
    :return: None
    """
    # Create folder to store image face
    path = os.path.join(os.getcwd(), "app", "static", "dataset", name)
    if not os.path.exists(path):
        os.makedirs(path)

    # Create camera object
    camera = cv2.VideoCapture(0)

    # Capture 100 image face
    count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()

        # Display the resulting frame
        cv2.imshow("Capture face", frame)

        # Save image face
        if cv2.waitKey(1) & 0xFF == ord("s"):
            count += 1
            cv2.imwrite(os.path.join(path, str(count) + ".jpg"), frame)
            print("Capture image face: ", count)

        # Quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Capture 100 image face
        if count >= 100:
            break

    # When everything done, release the capture
    camera.release()
    cv2.destroyAllWindows()


capture_face("HuyDN")
