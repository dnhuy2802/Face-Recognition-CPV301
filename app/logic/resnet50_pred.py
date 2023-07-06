import cv2
import numpy as np
from keras.models import load_model
import os
import dlib
from imutils import face_utils
from imutils import resize
from IPython.display import display

# Load the trained face recognition model
model = load_model(
    os.path.join(os.getcwd(), "app", "model",
                 "resnet50_face_recognition_model.h5"))
print("Loaded model from disk")

# Load the pre-trained face detection cascade classifier
print("[INFO] loading facial landmark predictor...")
# face_cascade = cv2.CascadeClassifier(
#     os.path.join(os.getcwd(), "app", "models",
#                  "haarcascade_frontalface_default.xml"))
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(
#     os.path.join(os.getcwd(), "app", "model",
#                  "shape_predictor_68_face_landmarks.dat"))

# Define the labels or names for each person in your dataset
train_data_dir = os.path.join(os.getcwd(), "app", "data", "train")
labels = [
    f for f in os.listdir(train_data_dir)
    if os.path.isdir(os.path.join(train_data_dir, f))
]

# Set the font and color for displaying the recognized face name
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 255, 0)  # Green

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

image_size = (224, 224)

while True:
    # Capture frame-by-frame from the camera
    ret, frame = video_capture.read()

    # flip image
    frame = cv2.flip(frame, 1)
    # frame = resize(frame, width=800)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    # faces = face_cascade.detectMultiScale(gray,
    #                                       scaleFactor=1.1,
    #                                       minNeighbors=5,
    #                                       minSize=(30, 30))
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    display(rects)

    # Loop over the detected faces
    for (i, rect) in enumerate(rects):
        # # Extract the face region of interest (ROI)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face = frame[y:y + h, x:x + w]

        # # Preprocess the face image (resize, normalize, etc.) before feeding it to the model
        # preprocessed_face = cv2.resize(
        #     face,
        #     image_size)  # Resize the face image to the desired input size
        # preprocessed_face = preprocessed_face.astype(
        #     'float32') / 255.0  # Normalize pixel values to the range of 0-1
        # preprocessed_face = np.expand_dims(preprocessed_face, axis=0)

        # Preprocess face
        # face = gray[y:y + h, x:x + w]
        preprocessed_face = cv2.resize(face, image_size)
        preprocessed_face = preprocessed_face.astype('float32') / 255.0
        preprocessed_face = np.expand_dims(preprocessed_face, axis=-1)
        preprocessed_face = np.expand_dims(preprocessed_face, axis=0)

        # Perform face recognition using the trained model
        # Pass the preprocessed face image to the model for prediction
        # You need to implement this part based on your model's input requirements

        # Get the predicted label (person name) and confidence score
        # You need to implement this part based on your model's output
        predictions = model.predict(preprocessed_face)

        # Get the predicted label (person name) and confidence score
        label_index = np.argmax(predictions[0])
        label = labels[label_index]
        confidence = predictions[0][label_index]

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        # bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the face name
        cv2.putText(frame, f"Face #{label} - ({confidence:.2f})",
                    (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
video_capture.release()
cv2.destroyAllWindows()
