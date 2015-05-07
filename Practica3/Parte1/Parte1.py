__author__ = 'Miguel'
import cv2
import glob
import os
import math
import KeyPoint
import Training
import Operations
import Processing
import numpy as np




TRAININGNUMBER = 33
PRROCESSINGNUMBER = 32
KPNUMBER = 150
operations = Operations.Operations()

#TRAINING
training = Training.Training()
training.train(TRAININGNUMBER,KPNUMBER)
arrayOwnKeyPoints = training.arrayOwnKeyPoints
globalDescArray = training.globalDescArray
flannArray = training.flannArray

#PROCESSING
processing = Processing.Processing(arrayOwnKeyPoints, globalDescArray, flannArray)
processing.process(PRROCESSINGNUMBER, KPNUMBER)




