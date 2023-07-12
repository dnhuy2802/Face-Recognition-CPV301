import numpy as np
from .pca import PCAFaces
from sklearn.svm import SVC
from ..utils.get_dataset import get_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


class PCASVM:
    def __init__(self, registed_faces: list[str], k: int = 5, split_ratio: float = 0.2):
        self.registed_faces = registed_faces
        self.split_ratio = split_ratio
        self.pca = PCAFaces(k=k)
        self.model = SVC(kernel='rbf', C=1000)
        self.scaler = MinMaxScaler()
        self.encoder = LabelEncoder()

    def __get_dataset(self):
        X, y = get_dataset(self.registed_faces)
        # Fit PCA
        X = self.pca.fit_faces(X)
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
        # Fit PCA
        pca_face = self.pca.fit_face(face)
        # Scale data
        pca_face = self.scaler.transform(pca_face)
        # Predict
        y_pred = self.model.predict(pca_face)
        # Decode label
        y_pred = self.encoder.inverse_transform(y_pred)
        return y_pred[0]
