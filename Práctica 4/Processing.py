__author__ = 'Edu'

import cv2
import Operations
import numpy as np

class Processing:

    def __init__(self):
        self.operations = Operations.Operations()

    def testing(self,file):

        C_MARGEN_SEGURIDAD =1.9

        #Lectura de la imagen y copias e la misma para sus futuros usos
        #
        imagen = cv2.imread(file, 0)
        imagenUmbralizada = self.operations.umbralizarImagen(imagen)
        pintarImagen = np.copy(imagen)
        imagenContorno = np.copy(imagenUmbralizada)


        contornos, jerarquiaContornos = cv2.findContours(imagenUmbralizada, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        rectangulos = self.operations.detectorMatriculas(imagen)

        #Pintar las matriculas donde se supone que esta en la imagen
        #
        if len(rectangulos) == 0:
            print("Matricula no detectada")
        else:
            matricula = rectangulos[0]

            #Aplicacion de filtros para eliminar todos los contornos fuera de la matricula y cuyo tamanho no sea el que
            # corresponde a un digito de una matricula
            #
            contornosEnMatricula = self.operations.filtrarContornos(contornos, matricula)



            if len(contornosEnMatricula) == 0:
                print("No se han detectado contornos en la matricula")
            else:
                #Pintamos los contornos en una copia de la imagen original para que podamos ver el resultado
                # con los contornos reconocidos por el sistema
                #
                self.operations.pintarContornos(pintarImagen, contornosEnMatricula)


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
                    vectorDeCaracteristicas = self.operations.transformarMatrizDeGrises(imagenContorno10)
                    matrizMuestras[fila][:] = vectorDeCaracteristicas[:]
                    fila += 1

            return matrizMuestras, pintarImagen