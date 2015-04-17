__author__ = 'Miguel'
import cv2
import glob
import os
import math
import KeyPoint
import numpy as np



arrayOwnKeyPoints = []
descArray = []
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
        orb = cv2.ORB(nfeatures=500, nlevels=4, scaleFactor=1.3)
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
        descArray.append(desA)
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

        print ""
        print "Imagen: ",file.title()
        I = cv2.imread(file, 0)  # Cargar imagen en blanco y negro
        totalMask = np.zeros(I.shape,np.uint8)
        for flann in flannArray:


            orb = cv2.ORB(nfeatures=500, nlevels=4, scaleFactor=1.3)
            kpA, desA = orb.detectAndCompute(I,None)
            ImageKp = cv2.drawKeypoints(I, kpA, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imshow("Processing kp", img2)
            # cv2.waitKey()

            matches = flann.knnMatch(desA,descArray[index],k=5)
            # print "------------- imagen -------------"
            matchesMask = [[0,0] for i in xrange(len(matches))]
            xImage,yImage = I.shape[:2]
            xImage = xImage/10
            yImage = yImage/10
            # Creamos la mascara donde iremos anotando los ptos que coinciden
            mask = np.zeros((xImage,yImage),np.uint8)

            for match in matches:
                for desc in match:
                    kp = kpA[desc.queryIdx]
                    trainingKp = arrayOwnKeyPoints[desc.imgIdx][desc.trainIdx]
                    distanteToCenterPolar = calcularCentro(kp, I)
                    




                    keyPoint = KeyPoint.KeyPoint(kp.angle, distanteToCenterPolar, kp.size, kp.pt)
                    trainResolution, kpoints = arrayOwnKeyPoints[index][:2]
                    # print "Nueva res", I.shape
                    # print "Antigua res", trainResolution
                   # print "entrenamiento nuestro", kpoints[desc.queryIdx].angle
                    #print "Procesmiento nuestro", keyPoint.angle

                    # Posicion en integer
                    x, y = kp.pt[:2]
                    pos = (int(x), int(y))

                    processingKp = KeyPoint.KeyPoint(kp.angle, distanteToCenterPolar, kp.size, pos)
                    #print "- training Position", trainingKp.position
                    #print "> processing Positionn", processingKp.position
                    #print "- training Angle", trainingKp.angle
                    #print "> processing Angle", processingKp.angle
                    #print "- training Size", trainingKp.size
                    #print "> processing Size", processingKp.size
                    #print "- training distanteToCenterPolar", trainingKp.distanceToCenter
                    #print "> processing distanteToCenterPolar", processingKp.distanceToCenter



                    #Rotamos el Kp y reescalamos su distancia hacia el centro
                    #   Reescalado: trainingKp.size/processingKp.size
                    #   Rotamos: ¿? Aplicar rotacion al angulo de rotacion para que vote a su centro

                    scale = trainingKp.size/processingKp.size
                    distCenterModule, distCenterVector, distCenterAngle, centerPt =  trainingKp.distanceToCenter [:4]
                    xVector, yVector=  distCenterVector[:2]
                    newX = xVector*scale
                    newY = yVector*scale


                    #processingdistCenterModule, processingdistCenterVector, processingdistCenterAngle, processingcenterPt =  processingKp.distanceToCenter [:4]

                    votingVector = (newX, newY)

                    #rotationAngle= processingdistCenterAngle - distCenterAngle

                    processingKpX,processingKpY = processingKp.position[:2]
                    voteX =  int((newX + processingKpX)/10)
                    voteY =  int((newY + processingKpY)/10)

                    #print "Antes. Voto: "
                    #print "X: ",voteX, " Y", voteY

                    #print "Angulo original: ", distCenterAngle
                    #print "Angulo procesamiento: ", processingdistCenterAngle

                    #Aplicacion de matriz de rotacion para rotar el vector de votacion
                    #voteX = voteX*math.cos(rotationAngle) + voteY*math.sin(rotationAngle)
                    #voteY = voteX* (-math.sin(rotationAngle)) + voteY * math.cos(rotationAngle)

                    #print "Despues. Voto: "
                    #print "X: ",voteX, " Y", voteY

                    vote = (voteX, voteY)

                    maskX, maskY = mask.shape[:2]
                    if(voteX<maskY and voteX>=0 and voteY<maskX and voteY>=0):
                        #Le damos la vuelta para que luego la imagen salga correcta
                        mask[voteY][voteX] += 1


                        # print "-----------------"



            #Una vez tengamos la mascara lo que hay que hacer es reescalarla para ver su resultado con resize (interopolando para no perder la forma)
            indexX = 0
            acumulativeMask = np.zeros(I.shape,np.uint8)
            xImage, yImage = I.shape[:2]
            acumulativeMask = cv2.resize(mask, (yImage, xImage), interpolation=cv2.INTER_NEAREST)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(acumulativeMask)
            #Normalizamos
            # acumulativeMask = acumulativeMask//maxVal*255
            # cv2.imshow("Processing normalized mask", acumulativeMask)

            totalMask = totalMask+acumulativeMask
            cv2.imshow("Processing image kp", ImageKp)

        index += 1
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(totalMask)
        cv2.imshow("Processing NOT normalized mask", totalMask)

        totalMask = totalMask//maxVal*255
        cv2.imshow("Processing normalized mask", totalMask)
        cv2.waitKey()

training()
processing()





