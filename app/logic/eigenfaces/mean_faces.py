import numpy as np
import cv2 as cv
from .constants import IMAGE_SIZE
from ...utils.get_path import get_images_abspath


class MeanFaces:
    def __init__(self, faces_path: str):
        self.faces_path = faces_path
        # Faces matrix
        # imgs_matrix shape: (m, n) where m is the number of images,
        # n is the number of pixels
        self.imgs_matrix: np.ndarray = np.array([])
        # Mean vector
        # mean_vector shape: (n, ) where n is the number of pixels
        self.mean_vector: np.ndarray = self.__compute_mean_vector()
        # Residuals (faces matrix - mean vector)
        # residuals shape: (m, n) where m is the number of images,
        # n is the number of pixels
        self.residuals: np.ndarray = self.__compute_residuals()

    def __compute_mean_vector(self) -> np.ndarray:
        # Get the absolute paths of the images
        abs_paths = get_images_abspath(self.faces_path)
        # Read the images
        images: list[np.ndarray] = [
            cv.imread(abs_path, cv.IMREAD_GRAYSCALE) for abs_path in abs_paths]
        # Create a numpy array of the images
        for image in images:
            flattened_image = image.flatten()
            # Add the flattened image to the matrix
            if self.imgs_matrix.size == 0:
                self.imgs_matrix = flattened_image
            else:
                self.imgs_matrix = np.vstack(
                    (self.imgs_matrix, flattened_image))
        # Compute the mean vector
        mean_vector = np.mean(self.imgs_matrix, axis=0)
        return mean_vector

    def __compute_residuals(self) -> np.ndarray:
        # Compute the residuals
        residuals = self.imgs_matrix - self.mean_vector
        return residuals

    def get_mean_face(self) -> np.ndarray:
        # Reshape the mean vector
        mean_face = self.mean_vector.reshape(*IMAGE_SIZE).astype(np.uint8)
        return mean_face

    def compute_eigenfaces(self, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        # Make sure k is less than the number of images
        m = self.residuals.shape[0]
        if k > m:
            raise ValueError(
                f'k: {k} must be less than or equal to the number of images: {m}')
        # Calculate covariance matrix
        # cov_mat shape: (m, m) where m is the number of images
        # (m, n) @ (n, m) = (m, m)
        cov_math = self.residuals @ self.residuals.T
        # Compute the eigenvalues and eigenvectors of the covariance matrix
        # eigenvalues shape: (m, )
        # eigenvectors shape: (m, m)
        eigenvalues, eigenvectors = np.linalg.eig(cov_math)
        # Sort the eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        # Sort the eigenvalues and eigenvectors
        eigenvalues = eigenvalues[idx]
        # Compute eigenfaces
        # eigenfaces shape: (n, m) where m is the number of images,
        # n is the number of pixels
        # (n, m) @ (m, m) = (n, m)
        eigenfaces = self.residuals.T @ eigenvectors
        # Normalize the eigenfaces
        # eigenfaces shape: (n, m) where m is the number of images,
        # n is the number of pixels
        eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)
        # Return the first k eigenvalues and eigenfaces
        return eigenvalues[:k], eigenfaces[:, :k]
