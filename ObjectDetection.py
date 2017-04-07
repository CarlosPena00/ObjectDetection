#ObjectDetection
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'Example')
import cart2PolarOpencv as HOG
import svm as sv

def getX(fromFile = 0,positive = 1):
    if positive == 1:
        VAR = "Data/positive/"
        CSV = "CSV/Positive_Samples.csv"
        NUM_OF_IMGS = 13233
        TYPEFILE = ".jpg"
    else:
        VAR = "Data/negative/"
        CSV = "CSV/Negative_Samples.csv"
        NUM_OF_IMGS = 453
        TYPEFILE = ".png"
    if fromFile == 0:        
        
        
        lista = []
        for i in range(1,NUM_OF_IMGS+1):#13233
            #Full image set (already cut, in ratio 1:1)
            src = cv2.imread(VAR+str(i)+TYPEFILE)
            src = cv2.pyrDown(src)
            src = cv2.pyrDown(src)
            histG = HOG.getHistogramOfGradients(src)
            histGN = HOG.blockNormalization(histG,histG)
            lista.append(histGN)
            #lista.extend(histG)        
            if i%int(NUM_OF_IMGS/100) == 0:
                print str(int( i/float(NUM_OF_IMGS) * 100))+"%"
        
        X = np.array(lista)
        np.savetxt(CSV,X,delimiter= ",",fmt="%.2f")
    else:
        dataset = pd.read_csv(CSV)
        X = dataset.iloc[:,:].values
    return X


XN = getX(fromFile = 1, positive= 0)
XP = getX(fromFile = 1, positive= 1)
YP = np.ones(shape=(13232,1),dtype = int)
YN = np.zeros(shape=(452,1), dtype = int)


X = np.vstack((XP,XN))
y = np.vstack((YP,YN))
Matrix = np.hstack((X,y))

X_train,X_test,y_train,y_test,y_pred,classifier,cm = sv.svm(X,y)


   

