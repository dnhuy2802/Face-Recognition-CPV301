import cv2
import numpy as np
from keras.models import load_model
import os
from IPython.display import display
from constants import *
import dlib

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

def detectFaceHogDlib(frame, face_detector, model):
    faceRects = face_detector(frame, 0)
    peoples = list()
    for faceRect in faceRects:
        # bounding box
        startX = faceRect.left()
        startY = faceRect.top()
        endX = faceRect.right()
        endY = faceRect.bottom()

        # expand the bounding box
        startX = int(startX - (endX - startX) * (PERCENT_EXPAND) / 2)
        startY = int(startY - (endY - startY) * (PERCENT_EXPAND) / 2)
        endX = int(endX + (endX - startX) * (PERCENT_EXPAND) / 2)
        endY = int(endY + (endY - startY) * (PERCENT_EXPAND) / 2)

        bboxes = [startX, startY, endX, endY]

        # detected face
        face = frame[startY:endY, startX:endX]
        face = cv2.resize(face, IMAGE_SIZE)
        face = np.expand_dims(face, axis=0)

        # predict the face
        pred = model.predict(face)
        confidence_face_pred = np.max(pred)

        # if the confidence_face_pred is higher than threshold
        if confidence_face_pred > CONF_FACE_THRESHOLD:
            # get the name of the face according to the predicted class
            name = class_names[np.argmax(pred)]
            peoples.append({
                "name": name,
                "bboxes": bboxes,
                "confidence": confidence_face_pred
            })
        else:
            name = "Unknown"
            peoples.append(
                {
                    "name": name,
                    "bboxes": bboxes,
                    "confidence": confidence_face_pred
                }
            )
    # return the face have confident highest and the bounding boxes
    if len(peoples) > 0:
        return peoples
    else:
        return None

def detectFaceOpenCVDnn(net, model, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300, 300),
        [104, 117, 123],
        False,
        False,
    )

    net.setInput(blob)
    detections = net.forward()
    peoples = list()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_THRESHOLD:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            # expand the bounding box
            x1 = int(x1 - (x2 - x1) * (PERCENT_EXPAND) / 2)
            y1 = int(y1 - (y2 - y1) * (PERCENT_EXPAND) / 2)
            x2 = int(x2 + (x2 - x1) * (PERCENT_EXPAND) / 2)
            y2 = int(y2 + (y2 - y1) * (PERCENT_EXPAND) / 2)

            bboxes = [x1, y1, x2, y2]

            # detected face
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, IMAGE_SIZE)
            face = np.expand_dims(face, axis=0)

            # predict the face
            pred = model.predict(face)
            confidence_face_pred = np.max(pred)

            # if the confidence_face_pred is higher than threshold
            if confidence_face_pred > CONF_FACE_THRESHOLD:
                # get the name of the face according to the predicted class
                name = class_names[np.argmax(pred)]
                peoples.append({
                    "name": name,
                    "bboxes": bboxes,
                    "confidence": confidence_face_pred
                })
            else:
                name = "Unknown"
                peoples.append(
                    {
                        "name": name,
                        "bboxes": bboxes,
                        "confidence": confidence_face_pred
                    }
                )
    # return the face have confident highest and the bounding boxes
    if len(peoples) > 0:
        return peoples
    else:
        return None

# Set the font and color for displaying the recognized face name
font = cv2.FONT_HERSHEY_SIMPLEX

class_names = [
    f for f in os.listdir(TRAIN_DATA_DIR)
    if os.path.isdir(os.path.join(TRAIN_DATA_DIR, f))
]
num_classes = len(class_names)
display(class_names)

# Load the trained face recognition model
model = load_model(os.path.join(MODEL_DIR,"model.keras"))
print("Loaded model from disk")

# Load the pre-trained face detection cascade classifier
print("[INFO] loading facial landmark predictor...")
modelFile = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
configFile = os.path.join(MODEL_DIR, "deploy.prototxt")
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

hogFaceDetector = dlib.get_frontal_face_detector()

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Unable to access the camera")
else:
    print("Access to the camera was successfully obtained")

print("Streaming started - to quit press ESC")
while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv2.flip(frame, 1)

    # detect peoples in the image
    peoples = detectFaceOpenCVDnn(net, model, frame)
    # peoples = detectFaceHogDlib(frame, hogFaceDetector, model)
    display(peoples)
    # loop over the peoples detected
    if peoples is not None:
        for people in peoples:
            name = people["name"]
            bboxes = people["bboxes"]
            confidence = people["confidence"]
            # draw the bounding box of the face along with the associatedq
            (x1, y1, x2, y2) = bboxes

            if name == "Unknown":
                color = RED
            else:
                color = GREEN

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "Face #{} - {:.2f}%".format(name, confidence*100), (x1 - 10, y1 - 10), font, FONT_SCALE, color, 1)

    # display the resulting frame
    cv2.imshow("Face detector - to quit press ESC", frame)

    # Exit with ESC
    key = cv2.waitKey(1)
    if key % 256 == 27:  # ESC code
        break
    # Exit with window close
    if cv2.getWindowProperty("Face detector - to quit press ESC", 0) < 0:
        break
    # Exit with q
    if key % 256 == ord('q'):
        break

# when everything done, release the capture
video_capture.release()
cv2.destroyAllWindows()
print("Streaming ended")
