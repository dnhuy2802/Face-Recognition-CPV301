import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from .lda import LDAFaces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import cv2 as cv
import numpy as np
from ...utils.get_path import get_images_abspath
from ..utils.split_image import split_img
from ..face_detectors.haar_cascade import HaarCascadeDetector

class KNN:
    def __init__(self,registed_faces: list[str], k: int = 5, split_ratio: float = 0.2):
        self.registed_faces = registed_faces
        self.split_ratio = split_ratio
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.scaler = MinMaxScaler()
        self.encoder = LabelEncoder()
        self.lda = LDAFaces(k=k)

    def get_dataset(self, registered_faces: list[str], path_only: bool = False) -> tuple[np.ndarray[cv.Mat | str], np.ndarray[str]]:
        # Init face detector
        face_detector = HaarCascadeDetector()
        # Init empty array for faces and labels
        faces = []
        labels = []
        # Get dataset
        for face in registered_faces:
            face_paths = get_images_abspath(face)
            for face_path in face_paths:
                if path_only:
                    faces.append(face_path)
                    labels.append(face)
                    continue
                img: cv.Mat = cv.imread(face_path, cv.IMREAD_GRAYSCALE)
                face_cordinate = face_detector.detect_face(img)
                face_image = split_img(img, face_cordinate)
                faces.append(face_image)
                labels.append(face)
        return np.array(faces), np.array(labels)

    def __get_dataset(self):
        X, y = self.get_dataset(self.registed_faces)
        # Fit LDA
        X = self.lda.fit_faces(X)
        return X, y
    
    def fit(self):
        X, y = self.__get_dataset()
        # Scale data
        X = self.scaler.fit_transform(X)

        # Encode labels
        y = self.encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Train model
        self.model.fit(X_train, y_train)
        # Evaluate model
        score = self.model.score(X_test, y_test)
        print(f'Accuracy: {score}')

    def predict(self, face: np.ndarray):
        # Flatten face
        face = face.flatten().reshape(1, -1)
        # Fit LDA
        lda_face = self.lda.fit_face(face)
        # Scale data
        lda_face = self.scaler.transform(lda_face)
        # Predict
        y_pred = self.model.predict(lda_face)
        # Decode label
        y_pred = self.encoder.inverse_transform(y_pred)
        return y_pred[0]


