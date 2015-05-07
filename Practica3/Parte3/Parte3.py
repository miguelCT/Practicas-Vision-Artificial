__author__ = 'Edu'

import cv2
import numpy as np
import cv2.cv as cv
import os
import glob



#Busca los rectangulos donde cree que esta la imagen
def detect(img, cascade,vecinos,size):
    rects = cascade.detectMultiScale(img ,scaleFactor=1.2, minNeighbors=vecinos, minSize=(size,size), flags=cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects



#Pinta los rectangulos
def draw_rects(img,rects,color):
    for x1,y1,x2,y2 in rects:
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)



def detectarCoches(file):

    #Fichero clasificador ya entrenado
    cascade_file = ("../haar/coches.xml")

    cascade = cv2.CascadeClassifier(cascade_file)

    cap = cv2.VideoCapture(file)
    print cap.isOpened()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(ret):
            # cv2.waitKey()
            rectangulos = detect(frame, cascade, 2, 120)
            vis = frame.copy()

            draw_rects(vis, rectangulos, (0, 255, 0))
            cv2.imshow('Detector de coches',vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.waitKey(1) & 0xFF == ord('p'):
                cv2.waitKey()

        else:
            break
    cap.release()
    cv2.destroyAllWindows()




def detectorMatriculas(file):
   #Fichero clasificador ya entrenado
    cascade_file = ("../haar/matriculas.xml")

    cascade = cv2.CascadeClassifier(cascade_file)

    cap = cv2.VideoCapture(file)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(ret):

            rectangulos = detect(frame, cascade,4,30)

            vis = frame.copy()

            draw_rects(vis, rectangulos, (0, 255, 0))
            cv2.imshow('Detector de matriculas',vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.waitKey(1) & 0xFF == ord('p'):
                cv2.waitKey()
        else:
            break
    cap.release()
    cv2.destroyAllWindows()



def main():


    os.chdir("../videos")
    for file in glob.glob("*.wmv"):
        detectarCoches(file)
        detectorMatriculas(file)


#Ejecucion principal
main()
