"""
- Using ConvNeXtBase tom train model
- This model is used to face recognition
- input: folder contain image face
- output: camera detect face and cover name of person
"""
import os
import numpy as np
import cv2
from IPython.display import display
from imutils import face_utils
import dlib

from keras.models import Sequential
from keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    Flatten,
)
from keras import utils
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.optimizers import SGD

# import EarlyStopping
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def ConvNeXtBase(input_shape: tuple[int], num_classes: int = 1):
    """
    ConvNeXtBase model
    :param input_shape: input shape of image
    :param classes: number of classes
    :return: model
    """
    model = Sequential()

    # Block 1
    model.add(
        Conv2D(
            32,
            kernel_size=(3, 3),
            strides=(1, 1),
            input_shape=input_shape,
            padding="same",
            activation="relu",
            kernel_constraint=maxnorm(3),
        ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(
        Conv2D(64,
               kernel_size=(5, 5),
               strides=(1, 1),
               activation="relu",
               padding="same",
               kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu", kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    return model


def load_data(path, image_size: tuple[int, int]):
    """
    Load data from folder
    :param path: path to folder
    :return: data
    """
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        os.path.join(os.getcwd(), "app", "models",
                     "shape_predictor_68_face_landmarks.dat"))

    img_data_list = []
    labels = []
    for inx, folder in enumerate(os.listdir(path)):
        for file in os.listdir(os.path.join(path, folder)):
            img = cv2.imread(os.path.join(path, folder, file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect faces in the grayscale frame
            rects = detector(gray, 0)
            for (i, rect) in enumerate(rects):
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, image_size)
                img_data_list.append(face)
                labels.append(inx)
            # img_data = cv2.resize(img, image_size)
            # img_data_list.append(img_data)
            # labels.append(inx)

    # Data Handle
    data = np.array(img_data_list)
    data = data.astype("float32")
    # scale down(so easy to work with)
    data /= 255.0
    data = np.expand_dims(data, axis=-1)

    # Encode labels
    labels = np.array(labels, dtype='int64')

    return data, labels


def train_model(path, image_size: tuple[int, int]):
    """
    Train model
    :param path: path to folder
    :return: model
    """
    # Load data
    data, labels = load_data(path, image_size)

    # Load labels
    num_classes = len(np.unique(labels))
    display(num_classes)

    # Convert labels to categorical
    Y = np_utils.to_categorical(labels, num_classes)

    # Suffle data
    x, y = shuffle(data, Y, random_state=42)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
    display(X_train.shape)
    display(X_test.shape)
    display(y_train.shape)
    display(y_test.shape)

    # get image shape
    img_shape = data[0].shape
    display(img_shape)

    # Create model
    model = ConvNeXtBase(input_shape=img_shape, num_classes=num_classes)

    # Compile model
    epochs = 25
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    # Fit the model with EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X_train,
              y_train,
              validation_data=(X_test, y_test),
              epochs=epochs,
              batch_size=32,
              verbose=2,
              callbacks=[early_stopping])

    # Model metrics Accuracy
    scores = model.evaluate(X_test, y_test, verbose=0)
    display("Accuracy: %.2f%%" % (scores[1] * 100))

    return model


path = os.path.join(os.getcwd(), "app", "resources")
image_size = (128, 128)
model = train_model(path, image_size)

# Save model
model.save(os.path.join(os.getcwd(), "app", "model", "model.h5"))

# # # Save model
# # model.save(os.path.join(path, "model_ai", "model.h5"))

# # serialize model to JSON
# model_json = model.to_json()
# with open(os.path.join(os.getcwd(), "app", "models", "model.json"),
#           "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights(os.path.join(os.getcwd(), "app", "models", "model.h5"))
# print("Saved model to disk")
