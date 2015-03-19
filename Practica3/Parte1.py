__author__ = 'Miguel'
import cv2
import glob
import os
import math
import KeyPoint

os.chdir("./training")

arrayOwnKeyPoints = []
descArray = []
kpArray = []
flannArray = []

def calcularCentro(kp, image):
    height, weight = image.shape[:2]
    centerY = height / 2
    centerX = weight / 2
    angleKp = kp.angle
    xKp, yKp = kp.pt[:2]

    module = math.sqrt(math.pow((centerX - xKp), 2) + math.pow((centerY - yKp), 2))
    if (centerY - yKp) != 0:
        angle = math.atan((centerX - xKp) / (centerY - yKp))
    else:
        angle = 0
    distanteToCenterPolar = (module, angle)
    return distanteToCenterPolar

#Entrenamiento con imágenes de prueba
def training ():
    for file in glob.glob("*.jpg"):
        I = cv2.imread(file, 0)  # Caragar imagen en grises
        orb = cv2.ORB(nfeatures=100, nlevels=4, scaleFactor=1.3)
        kpA, desA = orb.detectAndCompute(I,None)

        for kp in kpA:
            distanteToCenterPolar = calcularCentro(kp, I)
            keyPoint = KeyPoint.KeyPoint(kp.angle, distanteToCenterPolar, kp.size, kp.pt)
            arrayOwnKeyPoints.append(keyPoint)

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


#Procesamiento de las imágenes para detectar los coches
def processing():
    for flann in flannArray:
        for file in glob.glob("*.jpg"):

            I = cv2.imread(file, 0)  # Caragar imagen en grises
            orb = cv2.ORB(nfeatures=100, nlevels=4, scaleFactor=1.3)
            kpA, desA = orb.detectAndCompute(I,None)
            matches = flann.knnMatch(desA,descArray[0],k=2)
            print "------------- imagen -------------"
            print I
            print matches
            matchesMask = [[0,0] for i in xrange(len(matches))]

training()
processing()





