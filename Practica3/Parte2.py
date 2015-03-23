__author__ = 'Edu'

import cv2
import numpy as np
import cv2.cv as cv
import os
import glob




#Busca los rectangulos donde cree que esta la imagen
def detect(img, cascade):
    rects = cascade.detectMultiScale(img,scaleFactor=1.3, minNeighbors=4, minSize=(5,5), flags=cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects



#Pinta los rectangulos
def draw_rects(img,rects,color):
    for x1,y1,x2,y2 in rects:
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)



def main():

    #Fichero clasificador ya entrenado
    cascade_file = ("./haar/coches.xml")

    cascade = cv2.CascadeClassifier(cascade_file)

    os.chdir("./training")

    for file in glob.glob("*.jpg"):

        print("Imagen: ",file)
        img = cv2.imread(file,0)


        rects = detect(img, cascade)

        #Pintar los rectangulos donde se supone que esta la imagen
        for x1,y1,x2,y2 in rects:
            print (x1,y1,x2,y2)

        vis = img.copy()

        draw_rects(vis, rects, (0, 255, 0))

        for x1, y1, x2, y2 in rects:
            roi = img[y1:y2, x1:x2]
            vis_roi = vis[y1:y2, x1:x2]

        cv2.imshow('facedetect', vis)

    cv2.destroyAllWindows()



#Ejecucion principal
main()