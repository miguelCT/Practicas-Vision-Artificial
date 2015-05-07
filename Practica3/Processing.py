__author__ = 'Miguel'
__author__ = 'Miguel'
import cv2
import glob
import os
import KeyPoint
import Operations
import math
import numpy as np

class Processing:

    def __init__(self, arrayOwnKeyPoints, globalDescArray, flannArray):
        self.arrayOwnKeyPoints = arrayOwnKeyPoints
        self.globalDescArray = globalDescArray
        self.flannArray = flannArray
        self.operations = Operations.Operations()



    def processMatches (self, matches, processingImage, processingImageMask, kpProcessingArray):
        for match in matches:
            for desc in match:
                kp = kpProcessingArray[desc.queryIdx]
                trainingKp = self.arrayOwnKeyPoints[desc.imgIdx][desc.trainIdx]
                distanteToCenter = self.operations.calculateCenter(kp, processingImage)
                processingKp = KeyPoint.KeyPoint(kp.angle, distanteToCenter, kp.size, kp.pt)

                scale = trainingKp.size/processingKp.size
                distCenterModule, vector, distCenterAngle, centerPt =  trainingKp.distanceToCenter [:4]
                xVector, yVector =  vector[:2]
                xVectorScaled = xVector*scale
                yVectorScaled = yVector*scale

                rotationAngle = trainingKp.angle - processingKp.angle

                processingKpX,processingKpY = processingKp.position[:2]
                voteX = ((xVectorScaled + processingKpX)/10)
                voteY = ((yVectorScaled + processingKpY)/10)
                # Rotacion del vector de votacion
                voteXrotated = (voteX*math.cos(rotationAngle) - voteY*math.sin(rotationAngle))
                voteYrotated = (voteX*math.sin(rotationAngle) + voteY * math.cos(rotationAngle))

                # ATENCION: Con esto cancelas la rotacion
                voteXrotated = voteX
                voteYrotated = voteY

                maskX, maskY = processingImageMask.shape[:2]
                if(voteXrotated<maskY and voteXrotated>=0 and voteYrotated<maskX and voteYrotated>=0):
                    #Le damos la vuelta para que luego la imagen salga correcta
                    processingImageMask[int(voteYrotated)][int(voteXrotated)] += 1
                cv2.imshow("Processing mask in progress", processingImageMask*255)
                # cv2.waitKey()
        return processingImageMask

    #Procesamiento de las imagenes para detectar los coches
    def process(self, imageNum, kpNum):
        os.chdir("../processing")
        imageCont = 0

        for file in glob.glob("*.jpg"):
            index = 0
            if(imageCont==imageNum):
                break
            processingImage = cv2.imread(file, 0)  # Cargar imagen en blanco y negro
            finalMask = np.zeros(processingImage.shape, np.uint8)
            for flann in self.flannArray:
                orb = cv2.ORB(nfeatures=kpNum, nlevels=4, scaleFactor=1.3)
                kpProcessingArray, descProcessingArray = orb.detectAndCompute(processingImage,None)
                imageWithKp = cv2.drawKeypoints(processingImage, kpProcessingArray, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                xProcesingImage,yProcessingImage = processingImage.shape[:2]
                xProcesingImage = xProcesingImage/10
                yProcessingImage = yProcessingImage/10
                # Creamos la mascara donde iremos anotando los ptos que coinciden
                emptyImageMask = np.zeros((xProcesingImage,yProcessingImage), np.uint8)
                matches = flann.knnMatch(descProcessingArray, self.globalDescArray[index], k=5)
                processingImageMask = self.processMatches(matches, processingImage, emptyImageMask, kpProcessingArray)

                #Una vez tengamos la mascara lo que hay que hacer es reescalarla para ver su resultado con resize (interopolando para no perder la forma)
                # cv2.imshow("Processing image lines", processingImage)
                indexX = 0
                xImage, yImage = processingImage.shape[:2]
                processingImageMask = cv2.resize(processingImageMask, (yImage, xImage), interpolation=cv2.INTER_NEAREST)
                finalMask = finalMask+processingImageMask
                index += 1

            # cv2.imshow("Processing image kp", imageWithKp)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(finalMask)
            cv2.imshow("Processing NOT normalized mask", finalMask*255)
            finalMask = (finalMask/int(maxVal))*255
            cv2.imshow("Processing normalized mask", finalMask)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(finalMask)
            maxLocx, maxLocy = maxLoc[:2]
            pt2 = (maxLocx+200, maxLocy+100)
            pt1 = (maxLocx-200, maxLocy-100)
            cv2.rectangle(processingImage, pt1, pt2, 1111, thickness=2, lineType=8, shift=0)
            cv2.imshow("Processing with car", processingImage)
            cv2.waitKey()
            imageCont= imageCont+1