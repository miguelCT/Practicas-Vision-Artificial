__author__ = 'Edu'

import cv2
import numpy as np
import cv2.cv as cv
import os
import glob


#Fichero clasificador ya entrenado
cascade_file = ("../haar/coches.xml")
cascadeCars = cv2.CascadeClassifier(cascade_file)

#Fichero clasificador ya entrenado
cascade_file = ("../haar/matriculas.xml")
cascadeMatriculas = cv2.CascadeClassifier(cascade_file)


#Busca los rectangulos donde cree que esta la imagen
def detect(img, cascade,vecinos,size):
    rects = cascade.detectMultiScale(img,scaleFactor=1.2, minNeighbors=vecinos, minSize=(size,size), flags=cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects



#Pinta los rectangulos
def draw_rects(img,rects,color):
    for x1,y1,x2,y2 in rects:
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)


def detectarCoches():


    os.chdir("../processing")
    for file in glob.glob("*.jpg"):
        img = cv2.imread(file,1)

        rectangulos = detect(img, cascadeCars, 2, 120)

        vis = img.copy()

        draw_rects(vis, rectangulos, (0, 255, 0))

        cv2.imshow('Deteccion coches Haar', vis)
        cv2.waitKey()


def detectorMatriculas():


    os.chdir("../processing")

    for file in glob.glob("*.jpg"):
        img = cv2.imread(file,1)

        rectangulos = detect(img, cascadeMatriculas,4,30)

        vis = img.copy()

        draw_rects(vis, rectangulos, (0, 255, 0))

        cv2.imshow('Deteccion matriculas Haar', vis)
        cv2.waitKey()

def main():

    os.chdir("../training")
    print("DetectarCoches")
    detectarCoches()
    print("DetectarMatriculas")
    detectorMatriculas()
    cv2.destroyAllWindows()


#Ejecucion principal
main()
