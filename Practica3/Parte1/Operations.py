__author__ = 'Miguel'

import math


class Operations:
    def calculateCenter (self, keypoint, image):
        height, width = image.shape[:2]
        centerY = height / 2
        centerX = width / 2
        anglekeypoint = keypoint.angle
        xkeypoint, ykeypoint = keypoint.pt[:2]
        center = (centerX, centerY)
        #Este es el vector que une el keypoint con el centro. Vector de votacion
        xVector = centerX - xkeypoint
        yVector = centerY - ykeypoint
        vector = (xVector, yVector)

        module = int(math.sqrt(math.pow((centerX - xkeypoint), 2) + math.pow((centerY - ykeypoint), 2)))
        if (centerY - ykeypoint) != 0:
            angle = math.atan((centerX - xkeypoint) / (centerY - ykeypoint))
        else:
            angle = 0
        distanteToCenterPolar = (module, vector, angle, center)
        return distanteToCenterPolar