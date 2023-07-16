from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import cv2


class LDAFaces:
    def __init__(self, k: int = 5):
        self.lda = LDA(n_components=k, solver='svd')

    def fit_faces(self, faces: np.ndarray[cv2.Mat]) -> np.ndarray:
        img_stack = np.array([])
        for face in faces:
            img = np.array(face).flatten()
            if img_stack.size == 0:
                img_stack = img
            else:
                img_stack = np.vstack((img_stack, img))
        # Calculate LDA
        pca_faces = self.lda.fit_transform(img_stack)
        return pca_faces
    
    def fit_face(self, face: np.ndarray):
        return self.lda.transform(face)

    