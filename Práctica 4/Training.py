__author__ = 'Edu'

import glob
import os
import CommonOperations
import numpy as np
import cv2

class Training:
    def __init__(self):
        self.operations = CommonOperations.Operations()
    def training(self):
        os.chdir("./training_ocr")

        #Inicializacion de las estructuras necesarias
        #
        numeroFicheros = len([name for name in os.listdir("./")])
        matrizCaracteristicas = np.zeros((numeroFicheros, 100), np.int32)
        clases = np.zeros((numeroFicheros, 1), np.int32)
        indexMatrizCaracteristicas = 0


        for file in glob.glob("*.jpg"):

            #Lectura de la imagen y procesamiento para conseguir el vector de caracteristicas de la misma
            imagen = cv2.imread(file,0)
            imagenUmbralizada = self.operations.umbralizarImagen(imagen)
            imagenContorno = cv2.resize(imagenUmbralizada,(10,10))
            vectorDeCaracteristicas = self.operations.transformarMatrizDeGrises(imagenContorno)


            #Introducimos el nombre de la imagen que estamos procesando ,ya sea XXX o ESP, en las clases
            #
            nombre = str(file[0:3])

            if (nombre == 'ESP'):
                clases[indexMatrizCaracteristicas][0] = 0
            else:
                clases[indexMatrizCaracteristicas][0] = int(ord(file[0]))

            #Anhadimos el vector de caracteristicas a la matriz de caracteristicas
            #
            columna = 0
            for elemento in vectorDeCaracteristicas[0]:
                matrizCaracteristicas[indexMatrizCaracteristicas][columna] = vectorDeCaracteristicas[0][columna]
                columna += 1


            indexMatrizCaracteristicas += 1

        #Volvemos a cambiar al directorio raiz
        os.chdir("../")
        return matrizCaracteristicas, clases