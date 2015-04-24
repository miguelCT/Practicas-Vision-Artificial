__author__ = 'Karen'
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


os.chdir("./testing_ocr")
for file in glob.glob("*.jpg"):

    imagen = cv2.imread(file,0)
    imagenUmbralizada=cv2.adaptiveThreshold(imagen,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

    cv2.imshow("Imagen Umbralizada", imagenUmbralizada)

    imagenContornos = imagenUmbralizada
    contornos, jerarquiaContornos=cv2.findContours(imagenContornos, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    rectangulos = detectorMatriculas(imagen)


    #Pintar las matriculas donde se supone que esta en la imagen
    if len(rectangulos)==0 :
        print("Rectangulos vacios")
    else:
        matricula = rectangulos[0]
        x1Matricula,y1Matricula,x2Matricula,y2Matricula = matricula


    pintado = 0
    noPintado = 0

    #cv2.drawContours(imagenContornos, contornos, -1, (255, 255, 255))
    for contorno in contornos:
        x,y,w,h = cv2.boundingRect(contorno)
        x1Contorno = x
        y1Contorno = y
        x2Contorno = x+w
        y2Contorno = y+h
        #ratio = float(w)/h
        print("")
        print ("Matricula: ")
        print ("x: ",x1Matricula,"y: ",y1Matricula,"x2: ",x2Matricula,"y2 ",y2Matricula)
        print ("Contorno: ")
        print ("x: ", x1Contorno, "y: ", y1Contorno, "x2: ", x2Contorno, "y2 ", y2Contorno)



        cv2.rectangle(imagenContornos, (x1Matricula, y1Matricula), (x2Matricula, y2Matricula), (255, 255, 255))
        if (((x2Contorno-x1Contorno)< (y2Contorno-y1Contorno)) & ((x2Contorno-x1Contorno)> 10) & ((y2Contorno-y1Contorno) > 10)):
             if (x1Contorno >= x1Matricula) & (x2Contorno <=x2Matricula) & (y1Contorno >= y1Matricula) & (y2Contorno <= y2Matricula):
                pintado += 1
                #cv2.line(imagenContornos,(x1Matricula,y1Matricula),(x2Matricula,y2Matricula),color=(255, 255, 255))
                #cv2.line(imagenContornos,(x1Contorno,y1Contorno),(x2Contorno,y2Contorno),color=(255, 255, 255))
                cv2.rectangle(imagenContornos, (x1Contorno, y1Contorno), (x2Contorno, y2Contorno), (255, 255, 255))
        else:
            noPintado += 1

    print ("Pintado: ", pintado)
    print ("No Pintado: ", noPintado)
    print ("Total: ", pintado + noPintado)

    cv2.imshow("Contornos", imagenContornos)
    cv2.waitKey()