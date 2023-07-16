import os
import shutil
import cv2

import splitfolders
from IPython.display import display

# import constants
from deeplearning.constants import *

class DataSplit:
    def __init__(self):
        pass


    def center_crop(self, img, percent=0.5):
        """Returns center cropped image
        Args:
        img: image to be center cropped
        percent: percetage of image to be cropped
        """
        width, height = img.shape[1], img.shape[0]

        # process crop width and height for max available dimension
        crop_width = int(height * percent)
        crop_height = int(height * percent)

        # center_crop
        mid_x, mid_y = int(width / 2), int(height / 2)
        cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
        crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
        return crop_img


    def crop_image_from_center(self,data_path):
        """
        Crop image from center and save over the original image
        :param path: path to folder
        :return: None
        """
        for folder in os.listdir(data_path):
            for file in os.listdir(os.path.join(data_path, folder)):
                img = cv2.imread(os.path.join(data_path, folder, file))
                cropped_img = self.center_crop(img, percent=0.8)
                # Save new image in to temp folder
                cv2.imwrite(os.path.join(data_path, folder, file), cropped_img)
        display("Crop center complete!")


    def detectFaceOpenCVDnn(self, net, frame):

        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        blob = cv2.dnn.blobFromImage(
            image=frame,
            scalefactor=1.2,
            size=IMAGE_SIZE,
            swapRB=False,
            crop=False,
        )

        net.setInput(blob)
        detections = net.forward()
        bboxes = list()
        faces = list()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONF_THRESHOLD:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)

                # expand the bounding box
                x1 = int(x1 - (x2 - x1) * (PERCENT_EXPAND) / 2)
                y1 = int(y1 - (y2 - y1) * (PERCENT_EXPAND) / 2)
                x2 = int(x2 + (x2 - x1) * (PERCENT_EXPAND) / 2)
                y2 = int(y2 + (y2 - y1) * (PERCENT_EXPAND) / 2)

                bboxes.append([x1, y1, x2, y2])

                top = x1
                right = y1
                bottom = x2 - x1
                left = y2 - y1

                # detected face
                face = frame[right:right + left, top:top + bottom]
                # frame[right:right + face.shape[0], top:top + face.shape[1]] = face

                # append the face to the faces list
                faces.append(face)

        # return the face have confident highest and the bounding boxes
        if len(faces) > 0:
            # resize faces[0]
            faces[0] = cv2.resize(faces[0], IMAGE_SIZE)
            return faces[0], bboxes[0]
        else:
            return None, None

    def data_split(self):
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
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # save face after detect in to temp folder
        for folder in os.listdir(TEMP_DATA_DIR):
            for file in os.listdir(os.path.join(TEMP_DATA_DIR, folder)):
                img = cv2.imread(os.path.join(TEMP_DATA_DIR, folder, file))
                face, bboxes = self.detectFaceOpenCVDnn(net, img)
                if face is not None:
                    cv2.imwrite(os.path.join(TEMP_DATA_DIR, folder, file), face)
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

        display("Data split complete!")

data_split = DataSplit()
data_split.data_split()