__author__ = 'Karen,Edu,Miguel'

import cv2
import numpy as np
import glob
import os
import cv2.cv as cv
from sklearn.lda import LDA

#Busca los rectangulos donde cree que esta la imagen
def detect(img, cascade, vecinos, size):
    rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=vecinos, minSize=(size, size), flags=cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def detectorMatriculas(img):

    #Fichero clasificador ya entrenado
    cascade_file = ("../haar/matriculas.xml")

    cascade = cv2.CascadeClassifier(cascade_file)

    rectangulos = detect(img, cascade,3,30)

    return rectangulos

def pintarContornos (imagen,contornos):

    for contorno in contornos:
        x,y,w,h = cv2.boundingRect(contorno)
        x1Contorno = x
        y1Contorno = y
        x2Contorno = x+w
        y2Contorno = y+h

        cv2.rectangle(imagen, (x1Contorno, y1Contorno), (x2Contorno, y2Contorno), (0, 0, 0))
    return imagen

def filtrarContornos(contornos,matricula):

    C_TAM_MAXIMO_CONTORNO = 10

    contornosEnMatricula = []

    #Ordenamos los contornos tomando como referencia su ubicacion en el eje X
    contornos = ordenarContornos(contornos)

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

def ordenarContornos (listaContornos):
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

def umbralizarImagen(imagen):
    imagenUmbralizada = cv2.adaptiveThreshold(imagen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                              cv2.THRESH_BINARY_INV, 75, 10)
    return imagenUmbralizada

def transformarMatrizDeGrises(imagen):
    vectorDeCaracteristicas = np.zeros((1,100), np.int32)
    posicion = 0
    for fila in imagen:
        for elemento in fila:
            vectorDeCaracteristicas[0][posicion] = elemento
            posicion += 1

    return vectorDeCaracteristicas

class KNearest():
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.KNearest()

    def train(self, samples, responses):
        self.model = cv2.KNearest()
        self.model.train(samples, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, self.k)
        return results.ravel()

class NormalBayesClassifier():
    def __init__(self):
        self.model = cv2.NormalBayesClassifier()

    def train (self, samples, responses):
        self.model =cv2.NormalBayesClassifier()
        self.model.train(samples,responses)

    def predict (self, samples):
        retval,results = self.model.predict(samples)
        return results.ravel()

def evaluate_model(model, samples, tipo):
    resp = model.predict(samples)
    respuesta = []

    for elemento in resp:
        if elemento == 0:
            respuesta.append('ESP')
        else:
            respuesta.append(chr(elemento))

    print(tipo + ": " + ' '.join(respuesta))

def entrenarLDA(matrizCaracteristicas,clases):
    entrenadorLDA = LDA()
    entrenadorLDA.fit(matrizCaracteristicas, clases.ravel())
    return entrenadorLDA

def reducirDimensionalidad(entrenador,matriz):
    return np.ndarray.astype(entrenador.transform(matriz), np.float32)

def training():
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
        imagenUmbralizada = umbralizarImagen(imagen)
        imagenContorno = cv2.resize(imagenUmbralizada,(10,10))
        vectorDeCaracteristicas = transformarMatrizDeGrises(imagenContorno)


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

    return matrizCaracteristicas, clases

def testing(file):

    C_MARGEN_SEGURIDAD =1.9

    #Lectura de la imagen y copias e la misma para sus futuros usos
    #
    imagen = cv2.imread(file, 0)
    imagenUmbralizada = umbralizarImagen(imagen)
    pintarImagen = np.copy(imagen)
    imagenContorno = np.copy(imagenUmbralizada)


    contornos, jerarquiaContornos = cv2.findContours(imagenUmbralizada, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    rectangulos = detectorMatriculas(imagen)

    #Pintar las matriculas donde se supone que esta en la imagen
    #
    if len(rectangulos) == 0:
        print("Matricula no detectada")
    else:
        matricula = rectangulos[0]

        #Aplicacion de filtros para eliminar todos los contornos fuera de la matricula y cuyo tamaño no sea el que
        # corresponde a un digito de una matricula
        #
        contornosEnMatricula = filtrarContornos(contornos, matricula)



        if len(contornosEnMatricula) == 0:
            print("No se han detectado contornos en la matricula")
        else:
            #Pintamos los contornos en una copia de la imagen original para que podamos ver el resultado
            # con los contornos reconocidos por el sistema
            #
            pintarContornos(pintarImagen, contornosEnMatricula)


            matrizMuestras = np.zeros((len(contornosEnMatricula),100),np.float32)
            fila = 0

            #Obtenemos subimagenes de cada contorno para obtener su vector de caracteristicas
            #
            for contorno in contornosEnMatricula:

                x,y,w,h = cv2.boundingRect(contorno)
                x1Contorno = x
                y1Contorno = y
                x2Contorno = x+w
                y2Contorno = y+h


                imagenContornoRecortado = \
                    imagenContorno[(y1Contorno-C_MARGEN_SEGURIDAD - 0.7):(y2Contorno+C_MARGEN_SEGURIDAD + 0.7),
                    (x1Contorno - 0.5):(x2Contorno + C_MARGEN_SEGURIDAD)]

                imagenContorno10 = cv2.resize(imagenContornoRecortado, (10, 10))
                vectorDeCaracteristicas = transformarMatrizDeGrises(imagenContorno10)
                matrizMuestras[fila][:] = vectorDeCaracteristicas[:]
                fila += 1

        return matrizMuestras, pintarImagen

def main():

    matrizCaracteristicas, etiquetasClases = training()

    entrenador = entrenarLDA(matrizCaracteristicas,etiquetasClases)
    matrizCaracteristicasReducidas = reducirDimensionalidad(entrenador,matrizCaracteristicas)

    modelKNearest = KNearest(k=3)
    modelKNearest.train(matrizCaracteristicasReducidas, etiquetasClases)

    modelKNearest2 = KNearest(k=5)
    modelKNearest2.train(matrizCaracteristicasReducidas,etiquetasClases)

    modelKNearest3 = KNearest(k=7)
    modelKNearest3.train(matrizCaracteristicasReducidas,etiquetasClases)

    modelNormalBayesClassifier = NormalBayesClassifier()
    modelNormalBayesClassifier.train(matrizCaracteristicasReducidas,etiquetasClases)


    os.chdir("../testing_ocr")
    listaArchivos = glob.glob("*.jpg")
    index = 0
    C_SALTO = 1
    while (index < 40) & (index * C_SALTO < len(listaArchivos)):
        file = listaArchivos[index * C_SALTO]
        print(file)
        matriz_test, pintarImagen = testing(file)
        matriz_testReducida = reducirDimensionalidad(entrenador,matriz_test)

        cv2.imshow(file,pintarImagen)
        evaluate_model(modelKNearest, matriz_testReducida, "KNearest, k = " + str(modelKNearest.k))
        evaluate_model(modelKNearest2, matriz_testReducida, "KNearest, k = " + str(modelKNearest2.k))
        evaluate_model(modelKNearest3, matriz_testReducida, "KNearest, k = " + str(modelKNearest3.k))
        evaluate_model(modelNormalBayesClassifier, matriz_testReducida, "NormalBayesClassifier")
        print ""

        cv2.waitKey()

        index += 1


main()