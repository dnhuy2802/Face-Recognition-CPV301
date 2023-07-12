import cv2 as cv
import numpy as np
from sklearn.decomposition import PCA


class PCAFaces:
    def __init__(self, k: int = 5):
        self.pca = PCA(n_components=k, svd_solver='arpack')

    def fit_faces(self, faces: np.ndarray[cv.Mat]) -> np.ndarray:
        img_stack = np.array([])
        for face in faces:
            img = np.array(face).flatten()
            if img_stack.size == 0:
                img_stack = img
            else:
                img_stack = np.vstack((img_stack, img))
        # Calculate PCA
        pca_faces = self.pca.fit_transform(img_stack)
        return pca_faces

    def fit_face(self, face: np.ndarray):
        return self.pca.transform(face)
