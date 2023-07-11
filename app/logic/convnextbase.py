import os
import numpy as np
import tensorflow as tf
import warnings
warnings.simplefilter(action='ignore', category=Warning)

from keras.applications import ConvNeXtBase
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Layer, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.keras.applications.convnext import preprocess_input

from sklearn.metrics import classification_report
from IPython.display import display
from constants import *

# Define your custom layer
class CustomLayer(Layer):
    def __init__(self,num_class, activation='relu'):
        super(CustomLayer, self).__init__()
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation=activation)
        self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(264, activation=activation)
        self.dropout2 = Dropout(0.2)
        self.batchnorm = BatchNormalization()
        self.dense_num_class = Dense(num_class, activation='softmax')


    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.batchnorm(x)
        x = self.dense_num_class(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'flatten': self.flatten,
            'dense1': self.dense1,
            'dropout1': self.dropout1,
            'dense2': self.dense2,
            'dropout2': self.dropout2,
            'batchnorm': self.batchnorm,
            'dense_num_class': self.dense_num_class
        })
        return config

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

# Train the model
num_train_samples = len(train_generator.filenames)
num_validation_samples = len(validation_generator.filenames)

# get name of the train classes
train_labels = train_generator.class_indices
people_labels = list(train_labels.keys())
num_classes = len(people_labels)
display(people_labels)

# Load the pre-trained model
model_base = ConvNeXtBase(weights='imagenet',
                          include_top=False,
                          include_preprocessing=True,
                          input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the weights of the pre-trained layers
for layer in model_base.layers:
    layer.trainable = False

# Add additional layers on top of ConvNeXtBase
model = Sequential()
model.add(model_base)
model.add(GlobalAveragePooling2D())
# model custom layer
model.add(CustomLayer(num_classes))
# Compile the model
model.compile(optimizer=Adam(lr=LRATE, decay=DECAY, momentum=MOMENTUM),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss',
                               restore_best_weights=True,
                               patience=3,
                               verbose=1)
checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, "convnextbase_face_recognition_model.keras"),
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

# Save the trained model
model.save_weights(os.path.join(MODEL_DIR, "convnextbase_face_recognition_model.keras"), overwrite=True, save_format="keras")

# print classification report
test_generator.reset()
pred = model.predict(test_generator, verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)
labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Evaluate the model on the test data by Accuracy and Loss
scores = model.evaluate(test_generator, verbose=1)
print("Accuracy: %.2f%%" % (scores[1] * 100))
print("Loss: %.2f%%" % (scores[0] * 100))

report = classification_report(true_classes,
                               predicted_class_indices,
                               target_names=class_labels)

print(report)
