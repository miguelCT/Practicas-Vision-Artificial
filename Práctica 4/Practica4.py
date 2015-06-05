__author__ = 'Karen,Edu,Miguel'

import cv2
import numpy as np
import glob
import os
from sklearn.lda import LDA
import CommonOperations
import NormalBayesClassifier
import KNearest
import Training
import Testing
import EntrenadorLDA


def main():
    operations = CommonOperations.Operations()
    training = Training.Training()
    processing = Testing.Processing()
    matrizCaracteristicas, etiquetasClases = training.training()
    LDA = EntrenadorLDA.EntrenadorLDA()
    entrenadorLDA = LDA.entrenarLDA(matrizCaracteristicas,etiquetasClases)
    matrizCaracteristicasReducidas = LDA.reducirDimensionalidad(entrenadorLDA,matrizCaracteristicas)

    modelKNearest = KNearest.KNearest(k=3)
    modelKNearest.train(matrizCaracteristicasReducidas, etiquetasClases)

    modelKNearest2 = KNearest.KNearest(k=5)
    modelKNearest2.train(matrizCaracteristicasReducidas,etiquetasClases)

    modelKNearest3 = KNearest.KNearest(k=7)
    modelKNearest3.train(matrizCaracteristicasReducidas,etiquetasClases)

    modelNormalBayesClassifier = NormalBayesClassifier.NormalBayesClassifier()
    modelNormalBayesClassifier.train(matrizCaracteristicasReducidas,etiquetasClases)


    os.chdir("./testing_ocr")
    listaArchivos = glob.glob("*.jpg")
    for file in listaArchivos:
        print(file)
        matriz_test, pintarImagen = processing.testing(file)
        matriz_testReducida = LDA.reducirDimensionalidad(entrenadorLDA,matriz_test)

        cv2.imshow(file,pintarImagen)
        operations.evaluate_model(modelKNearest, matriz_testReducida, "KNearest, k = " + str(modelKNearest.k))
        operations.evaluate_model(modelKNearest2, matriz_testReducida, "KNearest, k = " + str(modelKNearest2.k))
        operations.evaluate_model(modelKNearest3, matriz_testReducida, "KNearest, k = " + str(modelKNearest3.k))
        operations.evaluate_model(modelNormalBayesClassifier, matriz_testReducida, "NormalBayesClassifier")
        print ""

        cv2.waitKey()

main()