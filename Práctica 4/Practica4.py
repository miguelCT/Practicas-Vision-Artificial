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
    testing = Testing.Testing()
    LDA = EntrenadorLDA.EntrenadorLDA()

    matrizCaracteristicas, etiquetasClases = training.training()
    entrenadorLDA = LDA.entrenarLDA(matrizCaracteristicas,etiquetasClases)
    matrizCaracteristicasReducidas = LDA.reducirDimensionalidad(entrenadorLDA,matrizCaracteristicas)

    modelKNearest = KNearest.KNearest(k=3)
    modelKNearest.train(matrizCaracteristicasReducidas, etiquetasClases)

    modelKNearest2 = KNearest.KNearest(k=5)
    modelKNearest2.train(matrizCaracteristicasReducidas, etiquetasClases)

    modelKNearest3 = KNearest.KNearest(k=7)
    modelKNearest3.train(matrizCaracteristicasReducidas, etiquetasClases)

    modelNormalBayesClassifier = NormalBayesClassifier.NormalBayesClassifier()
    modelNormalBayesClassifier.train(matrizCaracteristicasReducidas, etiquetasClases)


    os.chdir("./testing_full_system")
    listaArchivos = glob.glob("*.jpg")
    resultsFile = open('../results.txt', 'w')
    for file in listaArchivos:
        matriz_test = testing.testing(file)
        centroX, centroY = operations.detectorCentroCoche(file=file)
        if matriz_test != None:

            #Lo comentado se puede desbloquear para probar con otro clasificador diferente
            #Por defecto utilizamos KNearest con k = 5
            matriz_testReducida = LDA.reducirDimensionalidad(entrenadorLDA,matriz_test)
            # matricula = operations.evaluate_model(modelKNearest, matriz_testReducida)
            matricula = operations.evaluate_model(modelKNearest2, matriz_testReducida)
            # matricula = operations.evaluate_model(modelKNearest3, matriz_testReducida)
            # matricula = operations.evaluate_model(modelNormalBayesClassifier, matriz_testReducida)
        else:
            matricula = "NODETECTED"



        resultsLine = file + " " + str(centroX) + " " + str(centroY) + " " + matricula + "\n"
        resultsFile.write(resultsLine)


    resultsFile.close()
main()