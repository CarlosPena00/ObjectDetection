#ObjectDetection
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'Example')
import cart2PolarOpencv as HOG

VAR = "Data/positive/"
TYPEFILE = ".jpg"
lista = []
for i in range(1,13233):#13233
    src = cv2.imread(VAR+str(i)+TYPEFILE)
    src = cv2.pyrDown(src)
    src = cv2.pyrDown(src)
    histG = HOG.getHistogramOfGradients(src)
    lista.append(histG)
    if i%133 == 0:
        print str(int( i/13233.0 * 100))+"%"

