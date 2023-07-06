import os
import numpy as np
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import preprocess_input
from IPython.display import display
from sklearn.model_selection import train_test_split
import shutil


def data_split():
    # Set the path to your image data directory
    data_dir = os.path.join(os.getcwd(), "app", "resources")

    # Set the path to the directory where you want to save the train and test subsets
    output_dir = os.path.join(os.getcwd(), "app", "data")
    os.makedirs(output_dir, exist_ok=True)

    # Delete the existing train and test folders
    shutil.rmtree(os.path.join(output_dir, 'train'), ignore_errors=True)
    shutil.rmtree(os.path.join(output_dir, 'test'), ignore_errors=True)

    # Set the test size and random seed
    test_size = 0.2
    random_seed = 42

    # Get the list of all subdirectories (person folders) in the data directory
    person_folders = [
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f))
    ]

    # Iterate over each person folder
    for person_folder in person_folders:
        # Create subdirectories in the output directory for the person
        os.makedirs(os.path.join(output_dir, 'train', person_folder),
                    exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test', person_folder),
                    exist_ok=True)

        # Get the list of image filenames for the current person
        image_files = os.listdir(os.path.join(data_dir, person_folder))

        # Split the image filenames into train and test sets
        train_files, test_files = train_test_split(image_files,
                                                   test_size=test_size,
                                                   random_state=random_seed)

        # Move the train images to the train subdirectory
        for train_file in train_files:
            src = os.path.join(data_dir, person_folder, train_file)
            dst = os.path.join(output_dir, 'train', person_folder, train_file)
            shutil.copyfile(src, dst)

        # Move the test images to the test subdirectory
        for test_file in test_files:
            src = os.path.join(data_dir, person_folder, test_file)
            dst = os.path.join(output_dir, 'test', person_folder, test_file)
            shutil.copyfile(src, dst)

    display("Data split complete!")


data_split()

# Set the paths to your dataset
train_data_dir = os.path.join(os.getcwd(), "app", "data", "train")
validation_data_dir = os.path.join(os.getcwd(), "app", "data", "test")
person_folders = [
    f for f in os.listdir(train_data_dir)
    if os.path.isdir(os.path.join(train_data_dir, f))
]

# Number of unique individuals in your dataset
num_classes = len(person_folders)

# Load the pre-trained ConvNetBase model
conv_base = ResNet50(weights='imagenet',
                     include_top=False,
                     input_shape=(224, 224, 3))

# Freeze the weights of the pre-trained layers
for layer in conv_base.layers:
    layer.trainable = False

# Add additional layers on top of ResNet50
x = conv_base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model

# Specify the batch size and image dimensions
batch_size = 32
image_size = (224, 224)
epochs = 30
lrate = 0.01
decay = lrate / epochs

model = Model(inputs=conv_base.input, outputs=predictions)
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

# Compile the model
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation for training set
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rescale=1. / 255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Data augmentation for validation set
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, rescale=1. / 255)

# Load and prepare the training data
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=image_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

# Load and prepare the validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical')

# Train the model
num_train_samples = len(train_generator.filenames)
num_validation_samples = len(validation_generator.filenames)

# Fit the model with EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model.fit(train_generator,
          steps_per_epoch=num_train_samples // batch_size,
          epochs=epochs,
          validation_data=validation_generator,
          validation_steps=num_validation_samples // batch_size,
          callbacks=[early_stopping])

# Save the trained model
model.save(
    os.path.join(os.getcwd(), "app", "model",
                 "resnet50_face_recognition_model.h5"))

# Evaluate the model on the validation set
scores = model.evaluate(validation_generator, verbose=0)
display("Accuracy: %.2f%%" % (scores[1] * 100))
