__author__ = 'Karen,Edu,Miguel'

import cv2
import numpy as np
import glob
import os
import cv2.cv as cv
from sklearn.lda import LDA

C_TAM_MAXIMO_CONTORNO = 10


#Busca los rectangulos donde cree que esta la imagen
def detect(img, cascade, vecinos, size):
    rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=vecinos, minSize=(size, size), flags=cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

#Pinta los rectangulos
def draw_rects(img,rects,color):
    x1,y1,x2,y2 = rects
    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)

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

        cv2.rectangle(imagen, (x1Contorno, y1Contorno), (x2Contorno, y2Contorno), (255, 255, 255))

def filtrarContornos(contornos,matricula):
    contornosEnMatricula = []

    #Ordenamos los contornos tomando como referencia su ubicacion en el eje X
    contornos = ordenarContornos(contornos)

    for contorno in contornos:
        x,y,w,h = cv2.boundingRect(contorno)
        x1Contorno = x
        y1Contorno = y
        x2Contorno = x+w
        y2Contorno = y+h

        x1Matricula,y1Matricula,x2Matricula,y2Matricula = matricula

        ladoXContorno = x2Contorno-x1Contorno
        ladoYContorno = y2Contorno-y1Contorno

        #Verficar que es alto que ancho y que tiene mas de 10 pixeles de alto
        if (ladoXContorno < ladoYContorno) & (ladoYContorno > C_TAM_MAXIMO_CONTORNO) and \
            (x1Contorno >= x1Matricula) & (x2Contorno <= x2Matricula) & \
            (y1Contorno >= y1Matricula) & (y2Contorno <= y2Matricula):

            #Eliminar contornos duplicados
            if (len (contornosEnMatricula) != 0):
                contornoAnterior = contornosEnMatricula[len(contornosEnMatricula) -1]
                x2,y2,w2,h2 = cv2.boundingRect(contornoAnterior)

                if not (x2 + 2 >= x):
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
    imagenUmbralizada=cv2.adaptiveThreshold(imagen,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,75,10)
    return imagenUmbralizada

def imprimirMatrizDeGrises(imagen):
    for fila in imagen:
        for columna in fila:
            print '{:4}'.format(columna),
        print

def transformarMatrizDeGrises(imagen):
    vectorDeCaracteristicas = np.zeros((1,100), np.int32)
    posicion = 0
    for fila in imagen:
        for elemento in fila:
            vectorDeCaracteristicas[0][posicion] = elemento
            posicion += 1

    return vectorDeCaracteristicas

def testing(file):

    C_MARGEN_SEGURIDAD =1.5
    imagen = cv2.imread(file, 0)

    imagenUmbralizada = umbralizarImagen(imagen)
    contornoEnImagen = np.copy(imagenUmbralizada)
    matriculaEnImagen = np.copy(imagenUmbralizada)
    imagenContornos = np.copy(imagenUmbralizada)
    imagenContorno = np.copy(imagenUmbralizada)

    #cv2.imshow("Imagen Umbralizada", imagenUmbralizada)

    contornos, jerarquiaContornos = cv2.findContours(imagenUmbralizada, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    rectangulos = detectorMatriculas(imagen)


    #Pintar las matriculas donde se supone que esta en la imagen

    if len(rectangulos) == 0:
        print("Rectangulos vacios")
    else:
        matricula = rectangulos[0]



        #BORRAR
        x1Matricula,y1Matricula,x2Matricula,y2Matricula = matricula
        cv2.rectangle(matriculaEnImagen, (x1Matricula, y1Matricula), (x2Matricula, y2Matricula), (255, 255, 255))

        #FIN BORRAR


        contornosEnMatricula = filtrarContornos(contornos, matricula)



        if len(contornosEnMatricula) == 0:
            print("Contornos vacios")
        else:
            pintarContornos(imagenContornos, contornosEnMatricula)

            #cv2.imshow("Contornos",imagenContornos)

            matrizMuestras = np.zeros((len(contornosEnMatricula),100),np.float32)
            fila = 0


            #contornosEnMatricula = ordenarContornosEnMatricula(contornosEnMatricula)

            for contorno in contornosEnMatricula:

                x,y,w,h = cv2.boundingRect(contorno)
                x1Contorno = x
                y1Contorno = y
                x2Contorno = x+w
                y2Contorno = y+h



                imagenContornoRecortado = \
                    imagenContorno[y1Contorno-C_MARGEN_SEGURIDAD:y2Contorno+C_MARGEN_SEGURIDAD, x1Contorno - C_MARGEN_SEGURIDAD + 0.5:x2Contorno+C_MARGEN_SEGURIDAD]


                imagenContorno300 = cv2.resize(imagenContornoRecortado,(300,300))

                cv2.rectangle(contornoEnImagen, (x1Contorno, y1Contorno), (x2Contorno, y2Contorno), (255, 0, 255))
                #cv2.imshow("POST: contornoEnImagen", contornoEnImagen)


                #cv2.imshow(("Contorno Recortado"), imagenContorno300)

                #cv2.waitKey()

                imagenContorno10 = cv2.resize(imagenContornoRecortado, (10, 10))
                #cv2.imshow(("Contorno Recortado 10x10"), imagenContorno)





                vectorDeCaracteristicas = transformarMatrizDeGrises(imagenContorno10)
                matrizMuestras[fila][:] = vectorDeCaracteristicas[:]
                fila += 1
        return matrizMuestras

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

class EM():
    def __init__(self):
        self.model = cv2.EM()

    def train (self, samples, responses):
        self.model =cv2.EM()
        self.model.train(samples,responses)

    def predict (self, samples):
        resultados = []
        x,y = samples.shape
        matriz = np.zeros((1,y),np.float32)
        for sample in samples:
            matriz[0] = sample
            retval,results = self.model.predict(matriz)
            resultados.append(results)
        return resultados

def evaluate_model(model, digits, samples, labels,tipo):
    resp = model.predict(samples)
    respuesta = []

    for elemento in resp:
        if not (type(elemento) is np.ndarray):
            respuesta.append(chr(elemento))
        else:
            for elem in elemento:
                for item in elem:
                    respuesta.append(repr(item))
    print(tipo + ": " + ' '.join(respuesta))

def obtenerContornos(imagenUmbralizada):
    imagen = np.copy(imagenUmbralizada)
    contornos, jerarquiaContornos=cv2.findContours(imagenUmbralizada, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contornosEnNumero= []
    for contorno in contornos:
        x,y,w,h = cv2.boundingRect(contorno)
        x1Contorno = x
        y1Contorno = y
        x2Contorno = x+w
        y2Contorno = y+h

        contornosEnNumero.append(contorno)



    #pintarContornos(imagen,contornosEnNumero)


    if len(contornosEnNumero) != 0:
        contornosEnNumero = ordenarContornos(contornosEnNumero)
        contornoBueno = contornosEnNumero[0]

        x,y,w,h = cv2.boundingRect(contornoBueno)
        x1Contorno = x
        y1Contorno = y
        x2Contorno = x+w
        y2Contorno = y+h


        imagenContorno = imagen[y1Contorno:y2Contorno, x1Contorno:x2Contorno]

        imagenContorno10 = cv2.resize(imagenContorno,(10,10))

    else:
        imagenContorno10 = np.zeros((10,10),np.uint8)

    return imagenContorno10

def entrenarLDA(matrizCaracteristicas,clases):
    entrenadorLDA = LDA()
    entrenadorLDA.fit(matrizCaracteristicas, clases.ravel())
    return entrenadorLDA

def reducirDimensionalidad(entrenador,matriz):
    return np.ndarray.astype(entrenador.transform(matriz), np.float32)


def training():
    os.chdir("./training_ocr")

    numeroFicheros = len([name for name in os.listdir("./")])
    matrizCaracteristicas = np.zeros((numeroFicheros,100),np.int32)
    clases = np.zeros((numeroFicheros,1),np.int32)
    indexMatrizCaracteristicas = 0

    for file in glob.glob("*.jpg"):

        imagen = cv2.imread(file,0)

        imagenUmbralizada = umbralizarImagen(imagen)

        cv2.imshow("Imagen Umbralizada", imagenUmbralizada)

        # cv2.imshow("Imagen rescalada", imagenContorno)

        imagenContorno = obtenerContornos(imagenUmbralizada)

        if imagenContorno is None:
            imagenContorno = np.zeros((10, 10), np.int32)

        cv2.imshow("Imagen Contornos", imagenContorno)
        cv2.waitKey()
        vectorDeCaracteristicas = transformarMatrizDeGrises(imagenContorno)


        #etiquetas de cada clase. Tantas posiciones como filas en la matriz de caracteristicas
        clases[indexMatrizCaracteristicas][0] = int(ord(file[0]))

        #Add vector de caracteristicas a la matriz de caracteristicas
        columna = 0
        for elemento in vectorDeCaracteristicas[0]:
            matrizCaracteristicas[indexMatrizCaracteristicas][columna] = vectorDeCaracteristicas[0][columna]
            columna += 1

        indexMatrizCaracteristicas += 1

    return matrizCaracteristicas, clases



def main():

    matrizCaracteristicas, etiquetasClases = training()

    entrenador = entrenarLDA(matrizCaracteristicas,etiquetasClases)
    #matrizCaracteristicasReducidas = reducirDimensionalidad(entrenador,matrizCaracteristicas)
    matrizCaracteristicasReducidas = np.ndarray.astype(matrizCaracteristicas,np.float32)

    modelKNearest = KNearest(k=2)
    modelKNearest.train(matrizCaracteristicasReducidas, etiquetasClases)

    modelNormalBayesClassifier = NormalBayesClassifier()
    modelNormalBayesClassifier.train(matrizCaracteristicasReducidas,etiquetasClases)

    modelEM = EM()
    modelEM.train(matrizCaracteristicasReducidas,etiquetasClases)

    #os.chdir("../testing_ocr")
    listaArchivos = glob.glob("*.jpg")
    index = 0
    C_SALTO = 2
    while (index < 40) & (index * C_SALTO < len(listaArchivos)):
        file = listaArchivos[index * C_SALTO]
        print(file)
        #matriz_test = testing(file)
        imagen = cv2.imread(file, 0)
        imagenUmbralizada = umbralizarImagen(imagen)
        imagenUmbralizada10 = cv2.resize(imagenUmbralizada, (10,10))
        matriz_test = transformarMatrizDeGrises(imagenUmbralizada10)
        matriz_testReducida = np.ndarray.astype(matriz_test, np.float32)
        # matriz_testReducida = reducirDimensionalidad(entrenador,matriz_test)

        #imagen = cv2.imread(file, 0)
        cv2.imshow("Imagen",imagen)

        evaluate_model(modelKNearest, None, matriz_testReducida, etiquetasClases,"KNearest")
        evaluate_model(modelNormalBayesClassifier,None,matriz_testReducida,etiquetasClases,"NormalBayesClassifier")
        evaluate_model(modelEM,None,matriz_testReducida,etiquetasClases,"EM")

        cv2.waitKey()

        index += 1

main()