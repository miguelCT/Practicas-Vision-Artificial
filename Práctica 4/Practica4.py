__author__ = 'Karen,Edu,Miguel'
import cv2
import numpy as np
import glob
import os
import cv2.cv as cv

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
        if (((ladoXContorno) < (ladoYContorno)) & ((ladoXContorno) > 10) & ((ladoYContorno) > 10)) and \
            (x1Contorno >= x1Matricula) & (x2Contorno <= x2Matricula) & \
            (y1Contorno >= y1Matricula) & (y2Contorno <= y2Matricula):

            contornosEnMatricula.append(contorno)


    return contornosEnMatricula

def umbralizarImagen(imagen):
    imagenUmbralizada=cv2.adaptiveThreshold(imagen,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,75,10)
    return imagenUmbralizada

def testing():
    os.chdir("./testing_ocr")
    for file in glob.glob("*.jpg"):

        imagen = cv2.imread(file,0)

        imagenUmbralizada = umbralizarImagen(imagen)
        cv2.imshow("Imagen Umbralizada", imagenUmbralizada)


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

            if len(contornosEnMatricula) != 0:
                pintarContornos(imagenUmbralizada,contornosEnMatricula)
            else:
                print("Contornos vacios")

        cv2.imshow("Contornos", imagenUmbralizada)
        cv2.waitKey()

def training():
    os.chdir("../training_ocr")
    for file in glob.glob("*.jpg"):

        imagen = cv2.imread(file,0)

        imagenUmbralizada = umbralizarImagen(imagen)

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
            if (((ladoXContorno) < (ladoYContorno)) & ((ladoXContorno) > 10) & ((ladoYContorno) > 10)):
                contornosEnNumero.append(contorno)

        pintarContornos(imagenUmbralizada,contornosEnNumero)

        cv2.imshow("Imagen Umbralizada", imagenUmbralizada)
        cv2.waitKey()



def main():
    testing()
    training()

main()