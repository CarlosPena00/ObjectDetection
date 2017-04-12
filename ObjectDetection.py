#ObjectDetection
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import sys
from tqdm import tqdm
sys.path.insert(0, 'Example')
import HOG as HOG
import svm as sv


#Input will be (86,86,?)
def getX(fromFile = 0,positive = 1):
    if positive == 1:
        VAR = "Data/positive2/"
        CSV = "CSV/Positive_Samples.csv"
        NUM_OF_IMGS = 4964
        TYPEFILE = ".ppm"
    else:
        VAR = "Data/negative2/"
        CSV = "CSV/Negative_Samples.csv"
        NUM_OF_IMGS = 4675
        TYPEFILE = ".JPEG"
    if fromFile == 0:        
        lista = np.empty([1,2916])
        for i in tqdm(range(1,NUM_OF_IMGS+1)):#13233
            #Full image set (already cut, in ratio 1:1)
            src = cv2.imread(VAR+str(i)+TYPEFILE)
            src = cv2.resize(src,(86,86))
            histG = HOG.getHistogramOfGradients(src)            
            lista = np.vstack((lista,histG))
            #lista.extend(histG)        
           
        X = np.delete(lista,(0),axis=0)
        np.savetxt(CSV,X,delimiter= ",",fmt="%.2f")
    else:
        dataset = pd.read_csv(CSV)
        X = dataset.iloc[:,:].values
    return X

def train(classifier,stdScaler, std = 0):
    if std == 0:
        VAR = "Data/positive2/"
        NUM_OF_IMGS = 4964
        TYPEFILE = ".ppm"
        for i in tqdm(range(1,NUM_OF_IMGS+1)):#13233
            src = cv2.imread(VAR+str(i)+TYPEFILE)
            src = cv2.resize(src,(86,86))
            histG = HOG.getHistogramOfGradients(src)          
            histGE = stdScaler.transform(histG)
            print classifier.predict(histGE)
    else:
        VAR = "Example/test"
        TYPEFILE = ".jpg"
        src = cv2.imread(VAR+TYPEFILE)
        srcUp = src#cv2.pyrUp(src)
        rows,cols,channel = src.shape
        maxRows = rows/86
        maxCols = cols/86
        for j in tqdm(range(0,maxRows)):
            for i in range(0, maxCols):
                for dY in range(0,3):
                    for dX in range(0,3):                                
                        roi = HOG.getROIsrc(srcUp,j,i,px = 86,dy = dY*20, dx = dX*20)
                        rows,cols,channel = roi.shape
                        if rows != 86 or cols != 86:
                            roi = cv2.resize(roi,(86,86))
                        histG = HOG.getHistogramOfGradients(roi)
                        histGE = stdScaler.transform(histG)
                        if classifier.predict(histGE):
                            plt.imshow(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
                            plt.savefig(str(j*1000)+str(i*100)+str(dY*10)+str(dX)+"Foi"+".jpg")
        
if len(sys.argv) <=1:
    print "Error not flags: -c XN XP C -l C"
else:
    if sys.argv[1] == '-c':
        if sys.argv[2] == 'XN':
            XN = getX(fromFile = 0, positive= 0)
            YN = np.zeros(shape=(4674,1), dtype = int)
                    
        if sys.argv[2] == 'XP':
            XP = getX(fromFile = 0, positive= 1)
            YP = np.ones(shape=(4963,1),dtype = int)
        if sys.argv[2] == 'rbf':
            FILE_NAME = 'Model/model8kRBF.sav'
            XN = getX(fromFile = 1, positive= 0)
            YN = np.zeros(shape=(4674,1), dtype = int)
            XP = getX(fromFile = 1, positive= 1)
            YP = np.ones(shape=(4963,1),dtype = int)
            X = np.vstack((XP,XN))
            y = np.vstack((YP,YN))
            y = y.ravel()
            X_train,X_test,y_train,y_test,y_pred,classifier,cm,standardScaler = sv.svm(X,y,'rbf')
            pickle.dump(classifier, open(FILE_NAME, 'wb'))
            train(classifier,standardScaler,std = 1)
            
            
        if sys.argv[2] == 'linear':
            FILE_NAME = 'Model/model8kLinear.sav'
            XN = getX(fromFile = 1, positive= 0)
            YN = np.zeros(shape=(4674,1), dtype = int)
            XP = getX(fromFile = 1, positive= 1)
            YP = np.ones(shape=(4963,1),dtype = int)
            X = np.vstack((XP,XN))
            y = np.vstack((YP,YN))
            y = y.ravel()
            X_train,X_test,y_train,y_test,y_pred,classifier,cm,standardScaler = sv.svm(X,y,'linear')
            pickle.dump(classifier, open(FILE_NAME, 'wb'))
            
            
    if sys.argv[1] == '-l':
        if sys.argv[2] == 'rbf':
            FILE_NAME = 'Model/model8kRBF.sav'
            classifier = pickle.load(open(FILE_NAME, 'rb'))
            VAR = "Data/positive2/"
            NUM_OF_IMGS = 4964
            TYPEFILE = ".ppm"
            cont = 0
            for i in tqdm(range(1,NUM_OF_IMGS+1)):#13233
                src = cv2.imread(VAR+str(i)+TYPEFILE)
                src = cv2.resize(src,(86,86))
                histG = HOG.getHistogramOfGradients(src)            
                cont += classifier.predict(histG)
            print float(cont)/NUM_OF_IMGS
