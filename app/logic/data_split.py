import os
import shutil
from sklearn.model_selection import train_test_split
from IPython.display import display

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
    f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))
]

# Iterate over each person folder
for person_folder in person_folders:
    # Create subdirectories in the output directory for the person
    os.makedirs(os.path.join(output_dir, 'train', person_folder),
                exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', person_folder), exist_ok=True)

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
