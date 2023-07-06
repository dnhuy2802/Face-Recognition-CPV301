"""
- Capture 100 image face from camera using OpenCV
"""
import cv2
import os
import time
from imutils import face_utils
import dlib

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()


def capture_face(name: str):
    """
    Capture 50 image face from camera using OpenCV
    :param name: name of person
    :return: None
    """
    # Create folder to store image face
    path = os.path.join(os.getcwd(), "app", "resources", name)
    if not os.path.exists(path):
        os.makedirs(path)

    # Create camera object
    camera = cv2.VideoCapture(0)

    # Auto Capture 100 image face after 3s
    count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()

        # flip image
        frame = cv2.flip(frame, 1)
        # frame = imutils.resize(frame, width=800)

        # Display the resulting frame
        cv2.imshow("frame", frame)

        # Save image face
        cv2.imwrite(os.path.join(path, f"{name}_{count}.jpg"), frame)

        # Count
        count += 1

        # Stop when count = 50
        if count == 100:
            break

        # Stop when press 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Sleep 30ms
        time.sleep(0.05)

    # When everything done, release the capture
    camera.release()
    cv2.destroyAllWindows()


capture_face("HuyDN")
