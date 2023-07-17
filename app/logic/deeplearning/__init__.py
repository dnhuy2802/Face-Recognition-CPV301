import os
import shutil
import splitfolders
import cv2 as cv
from sklearn.model_selection import train_test_split
from keras.applications.convnext import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from ..utils.get_dataset import get_dataset
from .constants import *


class DeepLearningModelWrapper:
    def __init__(self,
                 registered_faces: list[str],
                 train_percent: int = DEFAULT_TRAIN_RATIO,
                 valid_percent: int = DEFAULT_VALID_RATIO,
                 test_percent: int = DEFAULT_TEST_RATIO,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 epochs: int = DEFAULT_EPOCHS,
                 fine_tune: int = DEFAULT_FINE_TUNE):
        self.registered_faces = registered_faces
        self.train_percent = train_percent
        self.valid_percent = valid_percent
        self.test_percent = test_percent
        self.batch_size = batch_size
        self.epochs = epochs
        self.fine_tune = fine_tune

    def __split_folders(self):
        # Copy the image data to the temp directory
        shutil.copytree(RESOURCES_DIR, TEMP_DATA_DIR, dirs_exist_ok=True)

        # Crop the images from the center
        self.crop_image_from_center(TEMP_DATA_DIR)

        # Delete the existing folders in the output directory
        shutil.rmtree(DATA_DIR, ignore_errors=True)

        # load face detection model
        modelFile = os.path.join(MODEL_DIR,
                                 "res10_300x300_ssd_iter_140000_fp16.caffemodel")
        configFile = os.path.join(MODEL_DIR, "deploy.prototxt")
        net = cv.dnn.readNetFromCaffe(configFile, modelFile)
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

        # save face after detect in to temp folder
        for folder in os.listdir(TEMP_DATA_DIR):
            for file in os.listdir(os.path.join(TEMP_DATA_DIR, folder)):
                img = cv.imread(os.path.join(TEMP_DATA_DIR, folder, file))
                face, bboxes = self.detectFaceOpenCVDnn(net, img)
                if face is not None:
                    cv.imwrite(os.path.join(
                        TEMP_DATA_DIR, folder, file), face)
                    # # resize the image to image_size
                    # img = cv2.imread(os.path.join(TEMP_DATA_DIR, folder, file))
                    # img = cv2.resize(img, IMAGE_SIZE)
                    # cv2.imwrite(os.path.join(TEMP_DATA_DIR, folder, file), img)
                else:
                    os.remove(os.path.join(TEMP_DATA_DIR, folder, file))

        splitfolders.ratio(input=TEMP_DATA_DIR,
                           output=DATA_DIR,
                           seed=42,
                           ratio=(0.8, 0.1, 0.1),
                           group_prefix=None)

        shutil.rmtree(TEMP_DATA_DIR, ignore_errors=True)

    def __data_generator(self):
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
                                                            batch_size=self.batch_size,
                                                            class_mode='categorical')

        validation_generator = validation_datagen.flow_from_directory(
            VALID_DATA_DIR,
            target_size=IMAGE_SIZE,
            batch_size=self.batch_size,
            class_mode='categorical')

        test_generator = test_datagen.flow_from_directory(TEST_DATA_DIR,
                                                          target_size=IMAGE_SIZE,
                                                          batch_size=self.batch_size,
                                                          class_mode='categorical')
