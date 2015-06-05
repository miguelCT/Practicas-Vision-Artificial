import cv2
from sklearn.lda import LDA
import numpy as np

class EntrenadorLDA:
        def __init__(self):
            self.model = LDA()

        def entrenarLDA(self, matrizCaracteristicas,clases):
            entrenadorLDA = self.model
            entrenadorLDA.fit(matrizCaracteristicas, clases.ravel())
            return entrenadorLDA

        def reducirDimensionalidad(self, entrenadorLDA,matriz):
            return np.ndarray.astype(entrenadorLDA.transform(matriz), np.float32)