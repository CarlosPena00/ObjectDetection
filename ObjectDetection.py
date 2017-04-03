#ObjectDetection
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'Example')
import cart2PolarOpencv as HOG

def getXPositive(fromFile = 0):
    if fromFile == 0:        
        NUM_OF_POSITIVE_IMG = 13233
        VAR = "Data/positive/"
        TYPEFILE = ".jpg"
        lista = []
        for i in range(1,NUM_OF_POSITIVE_IMG+1):#13233
            src = cv2.imread(VAR+str(i)+TYPEFILE)
            src = cv2.pyrDown(src)
            src = cv2.pyrDown(src)
            histG = HOG.getHistogramOfGradients(src)
            
            lista.append(histG)
            #lista.extend(histG)
            
            if i%133 == 0:
                print str(int( i/float(NUM_OF_POSITIVE_IMG) * 100))+"%"
        
        X = np.array(lista)
        np.savetxt("CSV/Positive_Samples.csv",X,delimiter= ",",fmt="%.2f")
    else:
        dataset = pd.read_csv("CSV/Positive_Samples.csv")
        X = dataset.iloc[:,:].values
    return X

X = getXPositive(1)
