__author__ = 'Karen'
import cv2
import numpy as np
import glob
import os

os.chdir("./testing_ocr")
for file in glob.glob("*.jpg"):

    imagen = cv2.imread(file,0)
    imagenUmbralizada=cv2.adaptiveThreshold(imagen,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

    cv2.imshow("Imagen Umbralizada", imagenUmbralizada)
    imagenContornos = imagenUmbralizada
    contornos, jerarquiaContornos=cv2.findContours(imagenContornos, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contorno in contornos:
        x,y,w,h = cv2.boundingRect(contorno)
        ratio = float(w)/h


        if (ratio > 1):
            print ("Pintado")
            print ratio
            cv2.rectangle(imagenContornos,(x,y),(x+w,y+h),(0,255,0))


    cv2.imshow("Contornos", imagenContornos)
    cv2.waitKey()