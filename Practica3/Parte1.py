__author__ = 'Miguel'
import cv2
import glob
import os
import math
import KeyPoint



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

    module = math.sqrt(math.pow((centerX - xKp), 2) + math.pow((centerY - yKp), 2))
    if (centerY - yKp) != 0:
        angle = math.atan((centerX - xKp) / (centerY - yKp))
    else:
        angle = 0
    distanteToCenterPolar = (module, angle)
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
            keyPoint = KeyPoint.KeyPoint(kp.angle, distanteToCenterPolar, kp.size, kp.pt)
            ownKP.append(keyPoint)

        arrayOwnKeyPoints.append((I.shape, ownKP))
        #Guardamos los kp y los descriptores de cada imagen de entrenamiento para su posterior uso
        descArray.append(desA)
        kpArray.append(kpA)

        img2 = cv2.drawKeypoints(I, kpA, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
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

            I = cv2.imread(file, 0)  # Caragar imagen en

            orb = cv2.ORB(nfeatures=100, nlevels=4, scaleFactor=1.3)
            kpA, desA = orb.detectAndCompute(I,None)
            matches = flann.knnMatch(desA,descArray[index],k=2)
            print "------------- imagen -------------"
            matchesMask = [[0,0] for i in xrange(len(matches))]
            for match in matches:
                for desc in match:
                    kp = kpA[desc.queryIdx]
                    distanteToCenterPolar = calcularCentro(kp, I)
                    keyPoint = KeyPoint.KeyPoint(kp.angle, distanteToCenterPolar, kp.size, kp.pt)
                    trainResolution, kpoints = arrayOwnKeyPoints[index][:2]
                    # print "Nueva res", I.shape
                    # print "Antigua res", trainResolution
                    print "Antiguo nuestro", kpoints[desc.queryIdx]
                    print "Nuevo nuestro", keyPoint


        index += 1

training()
processing()





