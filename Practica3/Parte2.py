__author__ = 'Edu'

import cv2
import numpy as np
import cv2.cv as cv
import os
import glob



#Busca los rectangulos donde cree que esta la imagen
def detect(img, cascade,vecinos,size):
    rects = cascade.detectMultiScale(img,scaleFactor=1.1, minNeighbors=vecinos, minSize=(size,size), flags=cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects



#Pinta los rectangulos
def draw_rects(img,rects,color):
    for x1,y1,x2,y2 in rects:
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)



def detectarCoches():

    #Fichero clasificador ya entrenado
    cascade_file = ("../haar/coches.xml")

    cascade = cv2.CascadeClassifier(cascade_file)

    for file in glob.glob("*.jpg"):

        print("Imagen: ",file)
        img = cv2.imread(file,0)

        rectangulos = detect(img, cascade,2,120)

        #Pintar los rectangulos donde se supone que esta la imagen
        if len(rectangulos)==0 :
            print("Rectangulos vacios")
        else:
            for x1,y1,x2,y2 in rectangulos:
                print (x1,y1,x2,y2)

        vis = img.copy()

        draw_rects(vis, rectangulos, (0, 255, 0))

        cv2.imshow('Deteccion coches Haar', vis)
        cv2.waitKey()




def detectorMatriculas():

    #Fichero clasificador ya entrenado
    cascade_file = ("../haar/matriculas.xml")

    cascade = cv2.CascadeClassifier(cascade_file)


    for file in glob.glob("*.jpg"):

        print("Imagen: ",file)
        img = cv2.imread(file,0)

        rectangulos = detect(img, cascade,4,30)

        #Pintar los rectangulos donde se supone que esta la imagen
        if len(rectangulos)==0 :
            print("Rectangulos vacios")
        else:
            for x1,y1,x2,y2 in rectangulos:
                print (x1,y1,x2,y2)

        vis = img.copy()

        draw_rects(vis, rectangulos, (0, 255, 0))

        cv2.imshow('Deteccion matriculas Haar', vis)
        cv2.waitKey()

def main():

    os.chdir("./training")
    print("DetectarCoches")
    detectarCoches()
    print("DetectarMatriculas")
    detectorMatriculas()
    cv2.destroyAllWindows()


#Ejecucion principal
main()
