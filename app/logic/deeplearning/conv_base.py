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
from keras.preprocessing.image import ImageDataGenerator
from keras.constraints import maxnorm
from keras.optimizers import Adam
# import EarlyStopping
from keras.callbacks import EarlyStopping, ModelCheckpoint
# import preprocess_input for Conv2D
from keras.applications.convnext import preprocess_input
from keras.initializers import RandomNormal, Constant
from constants import *


def ConvNeXtBase(num_classes: int = 1):
    """
    ConvNeXtBase model
    :param input_shape: input shape of image
    :param classes: number of classes
    :return: model
    """
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3),
               input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
               activation='relu',
               kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(64, (3, 3),
               activation='relu',
               kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def train_model(path):
    """
    Train model
    :param path: path to folder
    :return: model
    """

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

    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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
    
    train_labels = train_generator.class_indices
    people_labels = list(train_labels.keys())
    num_classes = len(people_labels)
    num_train_samples = len(train_generator.filenames)
    display(people_labels)

    # Create model
    model = ConvNeXtBase(num_classes=num_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=LRATE, decay=DECAY),
                  metrics=['accuracy'])

    # Define the checkpoint and earlystop callbacks
    early_stopping = EarlyStopping(monitor='val_loss',
                                restore_best_weights=True,
                                patience=3,
                                verbose=1)
    checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, "model.keras"),
                                monitor='val_loss',
                                save_best_only=True,
                                mode='min',
                                verbose=1)
    callbacks = [early_stopping, checkpoint]
    model.fit(train_generator,
          steps_per_epoch=num_train_samples // BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=validation_generator,
          callbacks=callbacks)

    # Model metrics Accuracy
    scores = model.evaluate(test_generator, verbose=1)
    display("Accuracy: %.2f%%" % (scores[1] * 100))

    return model

model = train_model(TRAIN_DATA_DIR)

# Save model
model.save(os.path.join(MODEL_DIR, "model.keras"), save_format="keras")
