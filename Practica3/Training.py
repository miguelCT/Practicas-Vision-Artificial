__author__ = 'Miguel'
import cv2
import glob
import os
import KeyPoint
import Operations

class Training:

    def __init__(self):
        self.arrayOwnKeyPoints = []
        self.globalDescArray = []
        self.flannArray = []
        self.operations = Operations.Operations()


    def trainingGetOwnKpFromImage(self, Image, kpArray):
        ownKP=[]
        for kp in kpArray:
            distanteToCenter =  self.operations.calculateCenter(kp, Image)
            keyPoint = KeyPoint.KeyPoint(kp.angle, distanteToCenter, kp.size, kp.pt)
            ownKP.append(keyPoint)
        return ownKP

    #Entrenamiento con imagenes de prueba
    def train (self, imageNum, kpNum):
        os.chdir("./training")
        imageCont = 0
        for file in glob.glob("*.jpg"):
            if(imageCont==imageNum):
                break
            I = cv2.imread(file, 0)  # Caragar imagen en grises
            orb = cv2.ORB(nfeatures=kpNum, nlevels=4, scaleFactor=1.3)
            kpA, desA = orb.detectAndCompute(I,None)
            ownKP=self.trainingGetOwnKpFromImage(I,kpA)
            # Guardamos los kpoint
            self.arrayOwnKeyPoints.append(ownKP)
            #Guardamos los kp y los descriptores de cada imagen de entrenamiento para su posterior uso
            self.globalDescArray.append(desA)
            I = cv2.drawKeypoints(I, kpA, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)
            search_params = dict()
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            flann.add(desA)
            #Guardamos los flann de cada imagen
            self.flannArray.append(flann)
            imageCont= imageCont+1