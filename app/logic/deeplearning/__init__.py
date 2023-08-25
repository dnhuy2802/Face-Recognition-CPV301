import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications.convnext import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from .vgg_model import VGGModel
from .resnet_model import ResNetModel
from .convnextbase_model import ConvNextBaseModel
from ..utils.get_dataset import get_dataset
from ...models.recognition_model import ModelType
from .utils import save_to_temp, get_labels
from .constants import *


class DeepLearningModelWrapper:
    def __init__(self,
                 registered_faces: list[str],
                 save_path: str,
                 type: str = ModelType.VGG19.value,
                 train_percent: int = DEFAULT_TRAIN_RATIO,
                 valid_percent: int = DEFAULT_VALID_RATIO,
                 test_percent: int = DEFAULT_TEST_RATIO,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 fine_tune: int = DEFAULT_FINE_TUNE):
        self.registered_faces = registered_faces
        self.save_path = save_path
        self.type = type
        self.train_percent = train_percent
        self.valid_percent = valid_percent
        self.test_percent = test_percent
        self.batch_size = batch_size
        self.fine_tune = fine_tune
        self.epochs = DEFAULT_EPOCHS
        self.model: Sequential = None
        self.labels = get_labels()

    def __split_folders(self):
        X, y = get_dataset(self.registered_faces, is_gray_scale=False)
        # Train, validation and test split
        X_val_train, X_test, y_val_train, y_test = train_test_split(
            X, y, test_size=self.test_percent, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_val_train, y_val_train, test_size=self.valid_percent / (1 - self.test_percent), random_state=42)
        # Write to temp folder
        for i, face in enumerate(y_train):
            save_to_temp(X_train[i], face, TRAIN_DATA_DIR)
        for i, face in enumerate(y_val):
            save_to_temp(X_val[i], face, VALID_DATA_DIR)
        for i, face in enumerate(y_test):
            save_to_temp(X_test[i], face, TEST_DATA_DIR)

    def __data_generator(self):
        # Data augmentation for training set, validation set and test set
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
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
        train_generator = train_datagen.flow_from_directory(
            TRAIN_DATA_DIR,
            target_size=INPUT_SHAPE[:2],
            batch_size=self.batch_size,
            class_mode='categorical')

        validation_generator = validation_datagen.flow_from_directory(
            VALID_DATA_DIR,
            target_size=INPUT_SHAPE[:2],
            batch_size=self.batch_size,
            class_mode='categorical')

        test_generator = test_datagen.flow_from_directory(
            TEST_DATA_DIR,
            target_size=INPUT_SHAPE[:2],
            batch_size=self.batch_size,
            class_mode='categorical')

        return train_generator, validation_generator, test_generator

    def __load_model(self):
        if self.type == ModelType.VGG19.value:
            model = VGGModel(
                self.registered_faces, self.save_path, self.batch_size, self.epochs, self.fine_tune)
        elif self.type == ModelType.RESNET50.value:
            model = ResNetModel(
                self.registered_faces, self.save_path, self.batch_size, self.epochs, self.fine_tune)
        elif self.type == ModelType.CONVNEXTBASE.value:
            model = ConvNextBaseModel(
                self.registered_faces, self.save_path, self.batch_size, self.epochs, self.fine_tune)
        return model.model_create()

    def fit(self):
        # Split data into train, validation and test
        self.__split_folders()
        # Get data generator
        train_generator, validation_generator, test_generator = self.__data_generator()
        # Load model
        model, callbacks = self.__load_model()
        # Train model
        try:
            model.fit(train_generator,
                      epochs=self.epochs,
                      validation_data=validation_generator,
                      callbacks=callbacks)
            score = model.evaluate(test_generator, verbose=1)
        except:
            score = [0, 0]
        # Evaluate model
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        # Delete temp folder
        for face in self.registered_faces:
            shutil.rmtree(os.path.join(TRAIN_DATA_DIR, face))
            shutil.rmtree(os.path.join(VALID_DATA_DIR, face))
            shutil.rmtree(os.path.join(TEST_DATA_DIR, face))
        # Return score
        self.model = model
        return score[1]

    def predict(self, face: np.ndarray):
        # Exapnd dimension
        face = np.expand_dims(face, axis=0)
        # Predict
        pred = self.model.predict(face)
        # Confidence
        confidence_face_pred = np.max(pred)
        # Return label
        if confidence_face_pred > CONF_FACE_THRESHOLD:
            return self.labels[np.argmax(pred)]
        else:
            return 'Unknown'

    def save(self, path: str):
        if self.type == ModelType.CONVNEXTBASE.value:
            self.model.save_weights(path)
        else:
            self.model.save(path)

    def load(self, path: str):
        if self.type == ModelType.CONVNEXTBASE.value:
            self.model.load_weights(path)
        else:
            self.model = load_model(path)
