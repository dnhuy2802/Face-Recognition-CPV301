"""
- Using VGG19 tom train model
- This model is used to face recognition
- input: folder contain image face
- output: camera detect face and cover name of person
"""
# Import necessary libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.callbacks import EarlyStopping
from IPython.display import display
from constants import *

import warnings

warnings.simplefilter(action='ignore', category=Warning)

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

# Load the VGG19 model without the top layer
model_base = VGG19(weights='imagenet',
                   include_top=False,
                   input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the weights of the pre-trained layers
for layer in model_base.layers:
    layer.trainable = False

# Add a custom top layer to the pre-trained model
model = Sequential()
model.add(model_base)
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation="tanh"))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='sigmoid'))

# Define the checkpoint and earlystop callbacks
early_stopping = EarlyStopping(monitor='val_loss',
                               restore_best_weights=True,
                               patience=3,
                               verbose=1)
checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, "vgg19_face_recognition_model.keras"),
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)
callbacks = [early_stopping, checkpoint]

# Compile the model
model.compile(optimizer=Adam(lr=LRATE, decay=DECAY),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator,
          steps_per_epoch=num_train_samples // BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=validation_generator,
          callbacks=callbacks)

# Save the trained model
model.save(os.path.join(MODEL_DIR,"vgg19_face_recognition_model.keras"), overwrite=True, save_format="keras")

# Evaluate the model on the test data
scores = model.evaluate(test_generator, verbose=1)
print("Accuracy: %.2f%%" % (scores[1] * 100))
