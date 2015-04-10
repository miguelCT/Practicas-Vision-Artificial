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
    pt = (centerX, centerY)
    #Este es eñ vector que une el kp con el centro. Vectpr de votacion
    xVector = centerX - xKp
    yVector = centerY - yKp
    vector = (xVector, yVector)

    module = int(math.sqrt(math.pow((centerX - xKp), 2) + math.pow((centerY - yKp), 2)))
    if (centerY - yKp) != 0:
        angle = math.atan((centerX - xKp) / (centerY - yKp))
    else:
        angle = 0
    distanteToCenterPolar = (module, vector, angle, pt)
    return distanteToCenterPolar

#Entrenamiento con imagenes de prueba
def training ():
    os.chdir("./training")
    for file in glob.glob("*.jpg"):
        I = cv2.imread(file, 0)  # Caragar imagen en grises
        orb = cv2.ORB(nfeatures=100, nlevels=4, scaleFactor=1.3)
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
            dCMdl, dCAgl, dCPt =  keyPoint.distanceToCenter[:3]
            cv2.line(I, keyPoint.position, dCPt, (255, 255, 0) , thickness=2, lineType=8, shift=0)

        # Guardamos los kpoint y la resolucion de la imagen
        arrayOwnKeyPoints.append((I.shape, ownKP))
        #Guardamos los kp y los descriptores de cada imagen de entrenamiento para su posterior uso
        descArray.append(desA)
        kpArray.append(kpA)
        I = cv2.drawKeypoints(I, kpA, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow("Result", I)
        # cv2.waitKey()
        # cv2.imshow("Result2", img2)
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
    for flann in flannArray:
        index = 0
        for file in glob.glob("*.jpg"):

            I = cv2.imread(file, 0)  # Caragar imagen en blanco y negro

            orb = cv2.ORB(nfeatures=100, nlevels=4, scaleFactor=1.3)
            kpA, desA = orb.detectAndCompute(I,None)
            I = cv2.drawKeypoints(I, kpA, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imshow("Processing kp", img2)
            # cv2.waitKey()

            matches = flann.knnMatch(desA,descArray[index],k=5)
            print "------------- imagen -------------"
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
                    #Despues compararemos el trainingKp connel processingKp de mas abajo (tsmaño, y hacer lansuma del vector de votscion)

                    distanteToCenterPolar = calcularCentro(kp, I)
                    # Posicion en integer
                    x, y = kp.pt[:2]
                    pos = (int(x), int(y))
                    processingKp = KeyPoint.KeyPoint(kp.angle, distanteToCenterPolar, kp.size, pos)
                    #Obtenemos el centro de la imagen
                    dCMdl, dCAgl, dCPt =  processingKp.distanceToCenter[:3]
                    cv2.line(I, processingKp.position, dCPt, (255, 255, 0) , thickness=2, lineType=8, shift=0)
                    trainResolution, kpoints = arrayOwnKeyPoints[index][:2]
                    x, y = processingKp.position[:2]
                    x = x/10
                    y = y/10
                    # Hay un problema ya que la imagen con la que creamos la mascara y la imagen de la que se han sacado los kpoint no es del mismo tamanno y por tanto no
                    # podremos usar la mascara como tal, habra que redimensionar
                    mask[x][y] += 1
                    # print "Antiguo nuestro", kpoints[desc.queryIdx]
                    # print "Nuevo nuestro", keyPoint

            print "Nueva resolucion", I.shape
            print "Antigua resolucion", trainResolution

            print mask
            cv2.imshow("Processing kp", I)
            cv2.imshow("Processing mask", mask)
            cv2.waitKey()
        index += 1

training()
processing()





