from imutils import face_utils
from imutils.face_utils import FaceAligner
import face_alignment
from IPython.display import display
from imutils import resize
import time
import cv2
import dlib
import os
import numpy as np

# Saving and loading model and weights
from keras.models import load_model


def recognize_face(frame, labels, model):
    pred_img = frame.astype("float32") / 255.0
    # Add fourth axis to image
    pred_img = np.expand_dims(pred_img, axis=-1)
    # Expand dimensions of image
    pred_img = np.expand_dims(pred_img, axis=0)

    query = model.predict(pred_img)
    best_match_index = np.argmax(query)
    display(labels)
    display(best_match_index)
    best_match_label = labels[best_match_index]
    confidence = query[0][best_match_index]

    if confidence > 0.5:
        return best_match_label, confidence
    else:
        best_match_label = "Unknown"
        confidence = 0
        return best_match_label, confidence


# load model from h5 file
loaded_model = load_model(os.path.join(os.getcwd(), "app", "model",
                                       "model.h5"))
print("Loaded model from disk")

# load labels
# labels = np.load(os.path.join(os.getcwd(), "app", "models", "labels.npy"))

# Load facial landmark predictor
print("[INFO] loading facial landmark predictor...")
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(
#     os.path.join(os.getcwd(), "app", "models",
#                  "shape_predictor_68_face_landmarks.dat"))

detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(
#     os.path.join(os.getcwd(), "app", "model",
#                  "shape_predictor_68_face_landmarks.dat"))

# Initialize face aligner
# fa = FaceAligner(predictor, desiredFaceWidth=256)
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
#                                   flip_input=False,
#                                   device='cpu',
#                                   face_detector='dlib')

# Create camera object
print("[INFO] camera sensor warming up...")
camera = cv2.VideoCapture(0)

people = []
for inx, folder in enumerate(
        os.listdir(os.path.join(os.getcwd(), "app", "resources"))):
    people.append(folder)

image_size = (112, 112)

while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Convert to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the face detections
    for (i, rect) in enumerate(rects):
        # Determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        # shape = predictor(gray, rect)
        # shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        # Preprocess face
        face = gray[y:y + h, x:x + w]
        # if face is not detected
        if face.size == 0:
            pred_img = cv2.resize(gray, image_size)
        else:
            pred_img = cv2.resize(face, image_size)
        face_lable, confidence = recognize_face(pred_img, people, loaded_model)
        # pred_img = pred_img.astype("float32") / 255.0
        # # Add fourth axis to image
        # pred_img = np.expand_dims(pred_img, axis=-1)
        # # Expand dimensions of image
        # pred_img = np.expand_dims(pred_img, axis=0)

        # Predict
        # Y_pred = loaded_model.predict(pred_img)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # # show the face name
        # if np.max(Y_pred) > 0.7:
        #     result = np.argmax(Y_pred, axis=1)
        #     name = people[result[0]]
        # else:
        #     name = "Unknown"
        cv2.putText(frame, f"Face #{face_lable} - ({confidence:.2f})",
                    (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        # # and draw them on the image
        # for (x, y) in shape:
        #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # show the frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Face Recognition', cv2.WND_PROP_VISIBLE) < 1:
        break

# do a bit of cleanup
cv2.destroyAllWindows()
camera.stop()
