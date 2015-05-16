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

    # print("")
    # print ("Matricula: ")
    # print ("x: ",x1Matricula,"y: ",y1Matricula,"x2: ",x2Matricula,"y2 ",y2Matricula)
    # print ("Contorno: ")
    # print ("x: ", x1Contorno, "y: ", y1Contorno, "x2: ", x2Contorno, "y2 ", y2Contorno)

    for contorno in contornos:
        x,y,w,h = cv2.boundingRect(contorno)
        x1Contorno = x
        y1Contorno = y
        x2Contorno = x+w
        y2Contorno = y+h

        x1Matricula,y1Matricula,x2Matricula,y2Matricula = matricula

        ladoXContorno = x2Contorno-x1Contorno
        ladoYContorno = y2Contorno-y1Contorno



        #Verficar que es alto que ancho y que tiene mas de 10 x 10 pixeles
        if (ladoXContorno < ladoYContorno) & (ladoYContorno > C_TAM_MAXIMO_CONTORNO) and \
            (x1Contorno >= x1Matricula) & (x2Contorno <= x2Matricula) & \
            (y1Contorno >= y1Matricula) & (y2Contorno <= y2Matricula):

            contornosEnMatricula.append(contorno)


    return contornosEnMatricula

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



def testing():
    os.chdir("./testing_ocr")
    numeroFicheros = len([name for name in os.listdir("./")])
    matrizMuestras = np.zeros((numeroFicheros,100),np.float32)
    fila = 0
    for file in glob.glob("*.jpg"):

        imagen = cv2.imread(file,0)

        imagenUmbralizada = umbralizarImagen(imagen)

        #cv2.imshow("Imagen Umbralizada", imagenUmbralizada)


        contornos, jerarquiaContornos=cv2.findContours(imagenUmbralizada, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        rectangulos = detectorMatriculas(imagen)


        #Pintar las matriculas donde se supone que esta en la imagen

        if len(rectangulos)== 0 :
            print("Rectangulos vacios")
        else:
            matricula = rectangulos[0]

            #BORRAR
            x1Matricula,y1Matricula,x2Matricula,y2Matricula = matricula
            cv2.rectangle(imagenUmbralizada, (x1Matricula, y1Matricula), (x2Matricula, y2Matricula), (255, 255, 255))

            #FIN BORRAR

            contornosEnMatricula=filtrarContornos(contornos,matricula)

            imagenContornos = np.copy(imagenUmbralizada)
            if len(contornosEnMatricula) == 0:
                print("Contornos vacios")
            else:
                pintarContornos(imagenContornos, contornosEnMatricula)

                #cv2.imshow("Contornos",imagenContornos)


                for contorno in contornosEnMatricula:

                    x,y,w,h = cv2.boundingRect(contorno)
                    x1Contorno = x
                    y1Contorno = y
                    x2Contorno = x+w
                    y2Contorno = y+h


                    imagenContorno = np.copy(imagen)
                    imagenContorno = imagenContorno[y1Contorno:y2Contorno, x1Contorno:x2Contorno]

                    imagenContorno = cv2.resize(imagenContorno,(300,300))

                    contornoEnImagen = np.copy(imagenUmbralizada)

                    imagenContorno = umbralizarImagen(imagenContorno)

                    cv2.rectangle(contornoEnImagen, (x1Contorno, y1Contorno), (x2Contorno, y2Contorno), (255, 0, 255))
                    #cv2.imshow("contornoEnImagen", contornoEnImagen)

                    #cv2.imshow(("Contorno Recortado"), imagenContorno)

                    #cv2.waitKey()

                    imagenContorno = cv2.resize(imagenContorno, (10, 10))
                    #cv2.imshow(("Contorno Recortado 10x10"), imagenContorno)





                    vectorDeCaracteristicas = transformarMatrizDeGrises(imagenContorno)
                    return vectorDeCaracteristicas
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



def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    print("evaluada: ", chr(resp))


def obtenerContornos(imagenUmbralizada):
    contornos, jerarquiaContornos=cv2.findContours(imagenUmbralizada, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for contorno in contornos:
        x,y,w,h = cv2.boundingRect(contorno)
        x1Contorno = x
        y1Contorno = y
        x2Contorno = x+w
        y2Contorno = y+h
        ladoXContorno = x2Contorno-x1Contorno
        ladoYContorno = y2Contorno-y1Contorno


        contornosEnNumero= []
        #Verficar que es alto que ancho y que tiene mas de 10 x 10 pixeles
        if ((ladoXContorno) < (ladoYContorno)) & ((ladoXContorno) > 10) & ((ladoYContorno) > 10):
            contornosEnNumero.append(contorno)



        #pintarContornos(imagenUmbralizada,contornosEnNumero)


        # cv2.imshow("Imagen Umbralizada con contornos", imagenUmbralizada)
        if len(contornosEnNumero) == 0:
            imagenContorno = imagenUmbralizada[y1Contorno:y2Contorno, x1Contorno:x2Contorno]
            imagenContorno = cv2.resize(imagenContorno,(10,10))
        else:
            imagenContorno = np.zeros((10,10),np.uint8)
        return imagenContorno

def reducirDimensionalidad(matrizCaracteristicas,clases):
    entrenadorLDA = LDA()
    entrenadorLDA.fit(matrizCaracteristicas, clases)
    matrizCaracReducidas64 = entrenadorLDA.transform(matrizCaracteristicas)


    matrizCaracReducidas32 = np.ndarray.astype(matrizCaracReducidas64, np.float32)
    return entrenadorLDA,matrizCaracReducidas32


def training():
    os.chdir("../training_ocr")

    numeroFicheros = len([name for name in os.listdir("./")])
    print "Archivos de training",numeroFicheros

    matrizCaracteristicas = np.zeros((numeroFicheros,100),np.int32)
    clases = np.zeros((numeroFicheros,1),np.int32)
    indexMatrizCaracteristicas = 0

    for file in glob.glob("*.jpg"):

        imagen = cv2.imread(file,0)

        imagenUmbralizada = umbralizarImagen(imagen)

        #cv2.imshow("Imagen Umbralizada", imagenUmbralizada)

        # cv2.imshow("Imagen rescalada", imagenContorno)

        imagenContorno = obtenerContornos(imagenUmbralizada)

        if imagenContorno is None:
            imagenContorno = np.zeros((10, 10), np.uint8)

        vectorDeCaracteristicas = transformarMatrizDeGrises(imagenContorno)


        #etiquetas de cada clase. Tantas posiciones como filas en la matriz de caracteristicas
        clases[indexMatrizCaracteristicas][0] = int(ord(file[0]))

        #Add vector de caracteristicas a la matriz de caracteristicas
        columna = 0
        for elemento in vectorDeCaracteristicas[0]:
            matrizCaracteristicas[indexMatrizCaracteristicas][columna] = vectorDeCaracteristicas[0][columna]
            columna += 1

        indexMatrizCaracteristicas += 1


    # X = np.array([[-1, -1],
    #               [-2, -1],
    #               [-3, -2],
    #               [1, 1],
    #               [2, 1],
    #               [3, 2]])
    #
    # y = np.array([1, 1, 2, 3, 6, 6])
    # clf = LDA()
    # clf.fit(X, y)
    #
    # print X.shape
    # print y
    # print(clf.predict([[-0.8, -1]]))

    entrenador, matrizCaracReducidas32 = reducirDimensionalidad(matrizCaracteristicas,clases)

    return entrenador,matrizCaracReducidas32, clases



def main():
    matriz_test = testing()
    entrenador, matrizCaracteristicasReducidas, etiquetasClases = training()

    matriz_entrenada = np.ndarray.astype(entrenador.transform(matriz_test), np.float32)
    model = KNearest(k=7)
    model.train(matrizCaracteristicasReducidas, etiquetasClases)

    evaluate_model(model, None, matriz_entrenada, etiquetasClases)


main()