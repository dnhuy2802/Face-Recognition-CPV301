"""
- Using VGG19 tom train model
- This model is used to face recognition
- input: folder contain image face
- output: camera detect face and cover name of person
"""
# Import necessary libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from IPython.display import display
from .constants import *

import os
import numpy as np
import cv2
import dlib
import warnings
warnings.simplefilter(action='ignore', category=Warning)


class VGGModel:
    def __init__(self):
        self.people_lables = list()
        self.num_classes = int()
        self.num_train_samples = int()

    def data_generator(self):
        # Data augmentation for training set, validation set and test set
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                           rotation_range=20,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           brightness_range=(0.7, 1),
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           vertical_flip=False,
                                           fill_mode='nearest')

        validation_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)
        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        # Load and prepare the training data, validation data and test data
        train_generator = train_datagen.flow_from_directory(TRAIN_DATA_DIR,
                                                            target_size=IMAGE_SIZE,
                                                            batch_size=BATCH_SIZE,
                                                            class_mode='categorical')

        validation_generator = validation_datagen.flow_from_directory(
            VALIDATION_DATA_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical')

        test_generator = test_datagen.flow_from_directory(TEST_DATA_DIR,
                                                          target_size=IMAGE_SIZE,
                                                          batch_size=BATCH_SIZE,
                                                          class_mode='categorical')

        # get name of the train classes
        train_labels = train_generator.class_indices
        self.people_lables = list(train_labels.keys())
        self.num_classes = len(self.people_lables)
        self.num_train_samples = len(train_generator.filenames)
        display(self.people_lables)
        display(self.num_classes)
        display(self.num_train_samples)

        return train_generator, validation_generator, test_generator

    def model_create(self):
        model_base = VGG16(weights='imagenet',
                           include_top=False,
                           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

        # Freeze the weights of the pre-trained layers
        for layer in model_base.layers:
            layer.trainable = False

        # Add a custom top layer to the pre-trained model
        model = Sequential()
        model.add(model_base)
        model.add(GlobalAveragePooling2D())
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.7))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation="tanh"))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(self.num_classes, activation='sigmoid'))

        # Define the checkpoint and earlystop callbacks
        early_stopping = EarlyStopping(monitor='val_loss',
                                       restore_best_weights=True,
                                       patience=3,
                                       verbose=1)
        checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, "vgg_model.keras"),
                                     monitor='val_loss',
                                     save_best_only=True,
                                     mode='min',
                                     verbose=1)
        callbacks = [early_stopping, checkpoint]

        # Compile the model
        model.compile(optimizer=Adam(lr=LRATE, decay=DECAY),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model, callbacks

    def model_train(self):
        # Create data generator
        train_generator, validation_generator, test_generator = self.data_generator()

        # Create the model
        model, callbacks = self.model_create()

        model.fit(train_generator,
                  steps_per_epoch=self.num_train_samples // BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=validation_generator,
                  callbacks=callbacks)

        # Save the trained model
        model.save(os.path.join(MODEL_DIR, "vgg_model.keras"),
                   overwrite=True, save_format="keras")

        # Evaluate the model on the test data
        scores = model.evaluate(test_generator, verbose=1)
        print("Accuracy: %.2f%%" % (scores[1] * 100))


class FaceRecognitionVGGModel:
    def __init__(self):
        self.people_lables = list()

    def convert_and_trim_bb(self, image, rect):
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

    def detectFaceHogDlib(self, frame, face_detector, model):
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
                name = self.people_lables[np.argmax(pred)]
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

    def detectFaceOpenCVDnn(self, net, model, frame):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        blob = cv2.dnn.blobFromImage(
            image=frame,
            scalefactor=1.2,
            size=IMAGE_SIZE,
            swapRB=False,
            crop=False,
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
                    name = self.people_lables[np.argmax(pred)]
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
        if len(peoples) > 0:
            return peoples
        else:
            return None

    def face_predict(self):
        self.people_lables = [f for f in os.listdir(
            TRAIN_DATA_DIR) if os.path.isdir(os.path.join(TRAIN_DATA_DIR, f))]

        # Load the model
        model = load_model(os.path.join(MODEL_DIR, "vgg_model.keras"))

        # Load the face detector
        modelFile = os.path.join(
            MODEL_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
        configFile = os.path.join(MODEL_DIR, "deploy.prototxt")
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # hogFaceDetector = dlib.get_frontal_face_detector()

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
            peoples = self.detectFaceOpenCVDnn(net, model, frame)
            # peoples = self.detectFaceHogDlib(frame, hogFaceDetector, model)
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
                    cv2.putText(frame, "Face #{} - {:.2f}%".format(name, confidence*100),
                                (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, 1)

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
