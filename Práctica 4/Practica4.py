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
    modelKNearest2.train(matrizCaracteristicasReducidas, etiquetasClases)

    modelKNearest3 = KNearest.KNearest(k=7)
    modelKNearest3.train(matrizCaracteristicasReducidas, etiquetasClases)

    modelNormalBayesClassifier = NormalBayesClassifier.NormalBayesClassifier()
    modelNormalBayesClassifier.train(matrizCaracteristicasReducidas, etiquetasClases)


    os.chdir("./testing_full_system")
    listaArchivos = glob.glob("*.jpg")
    resultsFile = open('../results.txt', 'w')
    for file in listaArchivos:
        print(file)
        matriz_test, pintarImagen = processing.testing(file)
        centroX, centroY = operations.detectorCentroCoche(file=file)
        print ("Centro: " + str(centroX) + " " + str(centroY))
        if matriz_test != None:
            matriz_testReducida = LDA.reducirDimensionalidad(entrenadorLDA,matriz_test)
            # matricula = operations.evaluate_model(modelKNearest, matriz_testReducida)
            # print ("KNearest, k " + str(modelKNearest.k) + " = " + matricula1)
            matricula = operations.evaluate_model(modelKNearest2, matriz_testReducida)
            # print ("KNearest, k " + str(modelKNearest2.k) + " = " + matricula2)
            # matricula = operations.evaluate_model(modelKNearest3, matriz_testReducida)
            # print ("KNearest, k " + str(modelKNearest3.k) + " = " + matricula)
            # matricula = operations.evaluate_model(modelNormalBayesClassifier, matriz_testReducida)
            # print ("NormalBayesClassifier " + " = " + matriculaN)
        else:
            matricula = "NODETECTED"



        resultsLine = file + " " + str(centroX) + " " + str(centroY) + " " + matricula + "\n"
        resultsFile.write(resultsLine)


    resultsFile.close()
main()