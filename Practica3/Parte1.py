__author__ = 'Miguel'
import cv2
import glob
import os
import math
import KeyPoint
import numpy as np



arrayOwnKeyPoints = []
globalDescArray = []
kpArray = []
flannArray = []

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


#Entrenamiento con imagenes de prueba
def training ():
    os.chdir("./training")
    for file in glob.glob("*.jpg"):
        I = cv2.imread(file, 0)  # Caragar imagen en grises
        orb = cv2.ORB(nfeatures=10, nlevels=4, scaleFactor=1.3)
        kpA, desA = orb.detectAndCompute(I,None)
        ownKP=[]
        for kp in kpA:
            distanteToCenterPolar = calcularCentro(kp, I)
            # Para obtener la posicion en integer
            x, y = kp.pt[:2]
            pos = (int(x), int(y))
            keyPoint = KeyPoint.KeyPoint(kp.angle, distanteToCenterPolar, kp.size, pos)
            ownKP.append(keyPoint)
            #Obtenemos el centro de la imagen
            dCMdl, dcVector, dCAgl, dCPt =  keyPoint.distanceToCenter[:4]
            cv2.line(I, keyPoint.position, dCPt, (255, 255, 0) , thickness=2, lineType=8, shift=0)

        # Guardamos los kpoint y la resolucion de la imagen
        arrayOwnKeyPoints.append(ownKP)
        #Guardamos los kp y los descriptores de cada imagen de entrenamiento para su posterior uso
        globalDescArray.append(desA)
        kpArray.append(kpA)
        I = cv2.drawKeypoints(I, kpA, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.imshow("Result", I)
        #cv2.waitKey()
        #cv2.imshow("Result2", img2)
        # cv2.waitKey()

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


#Procesamiento de las imagenes para detectar los coches
def processing():
    os.chdir("../processing")
    for file in glob.glob("*.jpg"):
        index = 0

        processingImage = cv2.imread(file, 0)  # Cargar imagen en blanco y negro
        finalMask = np.zeros(processingImage.shape, np.uint8)
        cv2.imshow("Processing empty finalMask", finalMask)

        for flann in flannArray:


            orb = cv2.ORB(nfeatures=10, nlevels=4, scaleFactor=1.3)
            kpProcessingArray, descProcessingArray = orb.detectAndCompute(processingImage,None)
            imageWithKp = cv2.drawKeypoints(processingImage, kpProcessingArray, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imshow("Processing kp", img2)
            # cv2.waitKey()

            matches = flann.knnMatch(descProcessingArray, globalDescArray[index], k=5)
            # print "------------- imagen -------------"
            xProcesingImage,yProcessingImage = processingImage.shape[:2]
            xProcesingImage = xProcesingImage/10
            yProcessingImage = yProcessingImage/10
            # Creamos la mascara donde iremos anotando los ptos que coinciden
            processinImageMask = np.zeros((xProcesingImage,yProcessingImage), np.uint8)
            cv2.imshow("Processing empty mask", processinImageMask)
            for match in matches:
                for desc in match:
                    kp = kpProcessingArray[desc.queryIdx]
                    trainingKp = arrayOwnKeyPoints[desc.imgIdx][desc.trainIdx]
                    distanteToCenterPolar = calcularCentro(kp, processingImage)
                    trainResolution, kpoints = arrayOwnKeyPoints[index][:2]
                    # Posicion en integer
                    x, y = kp.pt[:2]
                    pos = (int(x), int(y))

                    processingKp = KeyPoint.KeyPoint(kp.angle, distanteToCenterPolar, kp.size, pos)
                   #Rotamos el Kp y reescalamos su distancia hacia el centro
                    #   Reescalado: trainingKp.size/processingKp.size
                    #   Rotamos: Aplicar rotacion al angulo de rotacion para que vote a su centro

                    scale = trainingKp.size/processingKp.size
                    distCenterModule, distCenterVector, distCenterAngle, centerPt =  trainingKp.distanceToCenter [:4]
                    xVector, yVector =  distCenterVector[:2]
                    xVectorScaled = xVector*scale
                    yVectorScaled = yVector*scale

                    processingdistCenterModule, processingdistCenterVector, processingdistCenterAngle, processingcenterPt =  processingKp.distanceToCenter [:4]

                    votingVector = (xVectorScaled, yVectorScaled)

                    rotationAngle = trainingKp.angle - processingKp.angle


                    processingKpX,processingKpY = processingKp.position[:2]
                    voteX = int((xVectorScaled + processingKpX)/10)
                    voteY = int((yVectorScaled + processingKpY)/10)
                    #
                    print "Antes rot: "
                    print "X: ",voteX, " Y", voteY
                    #
                    print "Angulo kp testing: ", trainingKp.angle
                    print "Angulo kp procesamiento: ", processingKp.angle
                    print "Angulo rotacion: ", rotationAngle
                    #
                    # Rotacion del vector de votacion
                    voteXrotated = int(voteX*math.cos(rotationAngle) - voteY*math.sin(rotationAngle))
                    voteYrotated = int(voteX*math.sin(rotationAngle) + voteY * math.cos(rotationAngle))

                    print "Despues rot: "
                    print "X: ",voteXrotated, " Y", voteYrotated

                    vote = (voteXrotated, voteYrotated)
                    voteX10 = voteXrotated*10
                    voteY10 = voteYrotated*10
                    vote10 = (voteX10, voteY10)
                    cv2.line(processingImage, processingKp.position, vote10, (255, 255, 0) , thickness=2, lineType=8, shift=0)

                    maskX, maskY = processinImageMask.shape[:2]
                    if(voteXrotated<maskY and voteXrotated>=0 and voteYrotated<maskX and voteYrotated>=0):
                        #Le damos la vuelta para que luego la imagen salga correcta
                        processinImageMask[voteYrotated][voteXrotated] += 1


            #Una vez tengamos la mascara lo que hay que hacer es reescalarla para ver su resultado con resize (interopolando para no perder la forma)
            cv2.imshow("Processing image lines", processingImage)
            indexX = 0
            xImage, yImage = processingImage.shape[:2]
            processinImageMask = cv2.resize(processinImageMask, (yImage, xImage), interpolation=cv2.INTER_NEAREST)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(processinImageMask)
            #Normalizamos
            # acumulativeMask = acumulativeMask//maxVal*255
            # cv2.imshow("Processing normalized mask", acumulativeMask)

            finalMask = finalMask+processinImageMask
            cv2.imshow("Processing image kp", imageWithKp)

        index += 1
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(finalMask)
        cv2.imshow("Processing NOT normalized mask", finalMask)

        finalMask = finalMask//maxVal*255
        cv2.imshow("Processing normalized mask", finalMask)
        cv2.waitKey()

training()
processing()





