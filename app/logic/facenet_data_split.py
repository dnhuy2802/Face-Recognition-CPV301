import os
import cv2

from matplotlib import pyplot as plt
from PIL import Image
from numpy import savez_compressed, asarray
from IPython.display import display
from constants import *
from mtcnn.mtcnn import MTCNN

#Method to extract Face
def extract_image(image):
    img1 = Image.open(image)            #open the image
    img1 = img1.convert('RGB')          #convert the image to RGB format 
    pixels = asarray(img1)              #convert the image to numpy array
    detector = MTCNN()                  #assign the MTCNN detector
    f = detector.detect_faces(pixels)
    #fetching the (x,y)co-ordinate and (width-->w, height-->h) of the image
    x1,y1,w,h = f[0]['box']             
    x1, y1 = abs(x1), abs(y1)
    x2 = abs(x1+w)
    y2 = abs(y1+h)
    #locate the co-ordinates of face in the image
    store_face = pixels[y1:y2,x1:x2]
    plt.imshow(store_face)
    image1 = Image.fromarray(store_face,'RGB')    #convert the numpy array to object
    image1 = image1.resize((160,160))             #resize the image
    face_array = asarray(image1)                  #image to array
    return face_array

#Method to fetch the face
def load_faces(directory):
    faces = list()
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        face = extract_image(path)
        # face = cv2.imread(path)
        # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # face = cv2.resize(face, (160, 160))
        faces.append(face)
    return faces


#Method to get the array of face data(trainX) and it's labels(trainY)
def load_dataset(directory):
    x, y = [], []
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        #load all faces in subdirectory
        faces = load_faces(path)
        #create labels
        labels = [subdir for i in range(len(faces))]
        #summarize
        print("loaded %d sample for class: %s" % (len(faces),subdir) ) # print progress
        x.extend(faces)
        y.extend(labels)
    return asarray(x), asarray(y)

images = load_faces(os.path.join(TRAIN_DATA_DIR, "HuyDN"))[0]
plt.imshow(images)
plt.show()

# #load the datasets
# trainX, trainy = load_dataset(TRAIN_DATA_DIR)
# print(trainX.shape, trainy.shape)
# # load test dataset
# testX, testy = load_dataset(VALIDATION_DATA_DIR)
# print(testX.shape, testy.shape)
# #compress the data
# savez_compressed(os.path.join(DATA_DIR, 'face_dataset.npz'),trainX, trainy, testX, testy)
