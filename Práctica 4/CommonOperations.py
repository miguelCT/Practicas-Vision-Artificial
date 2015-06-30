__author__ = 'Edu'

import cv2
import cv2.cv as cv
import numpy as np

class Operations:

    #Busca los rectangulos donde cree que esta la imagen
    def detect(self,img, cascade, vecinos, size):
        rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=vecinos, minSize=(size, size), flags=cv.CV_HAAR_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        rects[:, 2:] += rects[:,:2]
        return rects


    def detectorCentroCoche(self,file):

        cascade_file = ("../haar/coches.xml")

        cascade = cv2.CascadeClassifier(cascade_file)
        img = cv2.imread(file)
        rectangulos = Operations.detect(self,img=img, cascade=cascade, vecinos=6, size= 120)

        if len(rectangulos)>0:
            x1, y1, x2, y2 = rectangulos[0]
            centro = (x2+x1)/2, (y2+y1)/2

            return centro

        else:
            return 0, 0

    def detectorMatriculas(self,img):

        #Fichero clasificador ya entrenado
        cascade_file = ("../haar/matriculas.xml")

        cascade = cv2.CascadeClassifier(cascade_file)

        rectangulos = Operations.detect(self,img=img, cascade=cascade, vecinos=4,size= 10)

        return rectangulos


    def filtrarContornos(self,contornos,matricula):

        C_TAM_MAXIMO_CONTORNO = 8

        contornosEnMatricula = []

        #Ordenamos los contornos tomando como referencia su ubicacion en el eje X
        contornos = Operations.ordenarContornos(self,contornos)

        for contorno in contornos:
            x,y,w,h = cv2.boundingRect(contorno)
            x1Contorno = x
            y1Contorno = y
            x2Contorno = x+w
            y2Contorno = y+h

            x1Matricula, y1Matricula, x2Matricula, y2Matricula = matricula

            ladoXContorno = x2Contorno-x1Contorno
            ladoYContorno = y2Contorno-y1Contorno

            #Verficar que es alto que ancho y que tiene mas de 10 pixeles de alto
            #
            if (ladoXContorno < ladoYContorno) & (ladoYContorno > C_TAM_MAXIMO_CONTORNO) and \
                                (x1Contorno >= x1Matricula) & (x2Contorno <= x2Matricula) & \
                                (y1Contorno >= y1Matricula) & (y2Contorno <= y2Matricula):

                #Eliminar contornos duplicados
                if (len (contornosEnMatricula) != 0):
                    contornoAnterior = contornosEnMatricula[len(contornosEnMatricula) -1]

                    x1ContornoAnterior, y1ContornoAnterior, wContornoAnterior, hContornoAnterior = cv2.boundingRect(contornoAnterior)

                    if not (x1ContornoAnterior + 2 >= x):

                        contornosEnMatricula.append(contorno)
                else:
                    contornosEnMatricula.append(contorno)

        return contornosEnMatricula

    def ordenarContornos (self,listaContornos):
        lista = []
        for elem in listaContornos:
            x,y,w,h = cv2.boundingRect(elem)
            lista.append(x)
        for i in range(len(lista)-1,0,-1):
            for j in range(i):
                if (lista[j]> lista[j+1]):
                    temp = lista [j]
                    lista[j] = lista [j+1]
                    lista[j+1] = temp

                    tempCont = listaContornos[j]
                    listaContornos[j] = listaContornos [j+1]
                    listaContornos[j+1] = tempCont
        return listaContornos

    def umbralizarImagen(self,imagen):
        imagenUmbralizada = cv2.adaptiveThreshold(imagen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                              cv2.THRESH_BINARY_INV, 75, 10)
        return imagenUmbralizada

    def transformarMatrizDeGrises(self,imagen):
        vectorDeCaracteristicas = np.zeros((1,100), np.int32)
        posicion = 0
        for fila in imagen:
            for elemento in fila:
                vectorDeCaracteristicas[0][posicion] = elemento
                posicion += 1

        return vectorDeCaracteristicas

    def evaluate_model(self ,model, samples):
        resp = model.predict(samples)
        respuesta = ""
        for elemento in resp:
            if elemento != 0:
                respuesta += (chr(elemento))

        return (respuesta)