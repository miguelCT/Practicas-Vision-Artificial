__author__ = 'Miguel'
import cv2
import glob
import os
import math
import KeyPoint
import numpy as np



arrayOwnKeyPoints = []
globalDescArray = []
flannArray = []
TRAININGNUMBER = 33
PRROCESSINGNUMBER = 10
KPNUMBER = 150

def calcularCentro(kp, image):
    height, width = image.shape[:2]
    centerY = height / 2
    centerX = width / 2
    angleKp = kp.angle
    xKp, yKp = kp.pt[:2]
    center = (centerX, centerY)
    #Este es el vector que une el kp con el centro. Vector de votacion
    xVector = centerX - xKp
    yVector = centerY - yKp
    vector = (xVector, yVector)

    module = int(math.sqrt(math.pow((centerX - xKp), 2) + math.pow((centerY - yKp), 2)))
    if (centerY - yKp) != 0:
        angle = math.atan((centerX - xKp) / (centerY - yKp))
    else:
        angle = 0
    distanteToCenterPolar = (module, vector, angle, center)
    return distanteToCenterPolar

def trainingGetOwnKpFromImage(Image, kpArray):
    ownKP=[]
    for kp in kpArray:
        distanteToCenter = calcularCentro(kp, Image)
        keyPoint = KeyPoint.KeyPoint(kp.angle, distanteToCenter, kp.size, kp.pt)
        ownKP.append(keyPoint)
    return ownKP

#Entrenamiento con imagenes de prueba
def training (imageNum):
    os.chdir("./training")
    imageCont = 0
    for file in glob.glob("*.jpg"):
        if(imageCont==imageNum):
            break
        I = cv2.imread(file, 0)  # Caragar imagen en grises
        orb = cv2.ORB(nfeatures=KPNUMBER, nlevels=4, scaleFactor=1.3)
        kpA, desA = orb.detectAndCompute(I,None)
        ownKP=trainingGetOwnKpFromImage(I,kpA)
        # Guardamos los kpoint
        arrayOwnKeyPoints.append(ownKP)
        #Guardamos los kp y los descriptores de cada imagen de entrenamiento para su posterior uso
        globalDescArray.append(desA)
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
        flannArray.append(flann)
        imageCont= imageCont+1



def processMatches (matches, processingImage, processingImageMask, kpProcessingArray):
    for match in matches:
        for desc in match:
            kp = kpProcessingArray[desc.queryIdx]
            trainingKp = arrayOwnKeyPoints[desc.imgIdx][desc.trainIdx]
            distanteToCenter = calcularCentro(kp, processingImage)
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
def processing(imageNum):
    os.chdir("../processing")
    imageCont = 0

    for file in glob.glob("*.jpg"):
        index = 0
        if(imageCont==imageNum):
            break
        processingImage = cv2.imread(file, 0)  # Cargar imagen en blanco y negro
        finalMask = np.zeros(processingImage.shape, np.uint8)
        for flann in flannArray:
            orb = cv2.ORB(nfeatures=KPNUMBER, nlevels=4, scaleFactor=1.3)
            kpProcessingArray, descProcessingArray = orb.detectAndCompute(processingImage,None)
            imageWithKp = cv2.drawKeypoints(processingImage, kpProcessingArray, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            xProcesingImage,yProcessingImage = processingImage.shape[:2]
            xProcesingImage = xProcesingImage/10
            yProcessingImage = yProcessingImage/10
            # Creamos la mascara donde iremos anotando los ptos que coinciden
            emptyImageMask = np.zeros((xProcesingImage,yProcessingImage), np.uint8)
            matches = flann.knnMatch(descProcessingArray, globalDescArray[index], k=5)
            processingImageMask = processMatches(matches, processingImage, emptyImageMask, kpProcessingArray)

            #Una vez tengamos la mascara lo que hay que hacer es reescalarla para ver su resultado con resize (interopolando para no perder la forma)
            # cv2.imshow("Processing image lines", processingImage)
            indexX = 0
            xImage, yImage = processingImage.shape[:2]
            processingImageMask = cv2.resize(processingImageMask, (yImage, xImage), interpolation=cv2.INTER_NEAREST)
            finalMask = finalMask+processingImageMask
            index += 1

        cv2.imshow("Processing image kp", imageWithKp)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(finalMask)
        cv2.imshow("Processing NOT normalized mask", finalMask*255)
        finalMask = (finalMask/int(maxVal))*255
        cv2.imshow("Processing normalized mask", finalMask)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(finalMask)
        cv2.circle(processingImage, maxLoc, 150, 11111, thickness=1, lineType=8, shift=0)
        cv2.imshow("Processing with car", processingImage)
        cv2.waitKey()
        imageCont= imageCont+1

training(TRAININGNUMBER)
processing(PRROCESSINGNUMBER)





