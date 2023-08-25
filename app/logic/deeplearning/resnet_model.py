# Import necessary libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, BatchNormalization
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint
from .constants import *
from .utils import get_optimizer, get_loss

import warnings
warnings.simplefilter(action='ignore', category=Warning)


class ResNetModel:
    def __init__(self, registered_faces: list[str], save_path: str, batch_size: int = DEFAULT_BATCH_SIZE, epochs: int = DEFAULT_EPOCHS, fine_tune: int = DEFAULT_FINE_TUNE):
        self.people_lables = registered_faces
        self.save_path = save_path
        self.num_classes = int()
        self.num_train_samples = int()
        self.batch_size = batch_size
        self.epochs = epochs
        self.fine_tune = fine_tune

    def model_create(self):
        model_base = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=INPUT_SHAPE)

        # Freeze the weights of the pre-trained layers
        if self.fine_tune > 0:
            for layer in model_base.layers[:-self.fine_tune]:
                layer.trainable = False
        else:
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
        checkpoint = ModelCheckpoint(self.save_path,
                                     monitor='val_loss',
                                     save_best_only=True,
                                     mode='min',
                                     verbose=1)
        callbacks = [early_stopping, checkpoint]

        # Compile the model
        model.compile(optimizer=get_optimizer(),
                      loss=get_loss(),
                      metrics=['accuracy'])

        return model, callbacks
