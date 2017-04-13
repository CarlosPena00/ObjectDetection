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

NUM_OF_IMGS_P = 4964
NUM_OF_IMGS_N = 4684
#Input will be (86,86,?)
def getX(fromFile = 0,positive = 1):

    if positive == 1:
        VAR = "Data/positive2/"
        CSV = "CSV/Positive_Samples.csv"
        NUM_OF_IMGS = NUM_OF_IMGS_P
        TYPEFILE = ".ppm"
    else:
        VAR = "Data/negative2/"
        CSV = "CSV/Negative_Samples.csv"
        NUM_OF_IMGS = NUM_OF_IMGS_N
        TYPEFILE = ".JPEG"
    if fromFile == 0:        
        lista = np.empty([1,2916])
        for i in tqdm(range(1,NUM_OF_IMGS+1)):#13233
            #Full image set (already cut, in ratio 1:1)
            src = cv2.imread(VAR+str(i)+TYPEFILE)
            rows,cols,channel = src.shape 
            if rows > 1 and cols > 1:
                src = cv2.resize(src,(86,86))
                histG = HOG.getHistogramOfGradients(src)
            else:
                histG = np.zeros(2916)
            lista = np.vstack((lista,histG))     
           
        X = np.delete(lista,(0),axis=0)
        np.savetxt(CSV,X,delimiter= ",",fmt="%.2f")
    else:
        dataset = pd.read_csv(CSV)
        X = dataset.iloc[:,:].values
    return X

def train(classifier,stdScaler, std = 0):
    if std == 0:
        VAR = "Data/positive2/"
        NUM_OF_IMGS = NUM_OF_IMGS_P
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
        src2 = src.copy()
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
                            cv2.rectangle(src2,((dX*20+86+(i-1)*86),(dY*20+86+(j-1)*86)),((dX*20+86+i*86),(dY*20+86+j*86)),(0,0,255))
                            plt.imshow(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
                            cv2.imwrite("Img/"+str(j*1000)+str(i*100)+str(dY*10)+str(dX)+"Foi"+".jpg",roi)
        cv2.imwrite("Rect.jpg",src2)
        
if len(sys.argv) <=1:
    print "Error not flags: -x n p || -c rbf rf linear || -l rbf rf linear"
else:
    if sys.argv[1] == '-x':
        if sys.argv[2] == 'n':
            XN = getX(fromFile = 0, positive= 0)
            rows, cols = XN.shape
            YN = np.zeros(shape=(rows,1), dtype = int)
                    
        if sys.argv[2] == 'p':
            XP = getX(fromFile = 0, positive= 1)
            rows, cols = XP.shape
            YP = np.ones(shape=(rows,1),dtype = int)
    if sys.argv[1] == '-c':
        
        XN = getX(fromFile = 1, positive= 0)
        rowsN, colsN = XN.shape
        YN = np.zeros(shape=(rowsN,1), dtype = int)
        
        XP = getX(fromFile = 1, positive= 1)
        rowsP, colsP = XP.shape
        YP = np.ones(shape=(rowsP,1),dtype = int)
        
        X = np.vstack((XP,XN))
        y = np.vstack((YP,YN))
        y = y.ravel()
        if sys.argv[2] == 'rbf':
            FILE_NAME = 'Model/model8kRBF.sav'
            FILE_NAME_SCALAR = 'Model/scalar8kRBF.sav'
            X_train,X_test,y_train,y_test,y_pred,classifier,cm,standardScaler = sv.svm(X,y,'rbf')
        if sys.argv[2] == 'rf':
            FILE_NAME = 'Model/model8kRF.sav'
            FILE_NAME_SCALAR = 'Model/scalar8kRF.sav'
            X_train,X_test,y_train,y_test,y_pred,classifier,cm,standardScaler = sv.svm(X,y,'linear')
        if sys.argv[2] == 'linear':
            FILE_NAME = 'Model/model8kLinear.sav'
            FILE_NAME_SCALAR = 'Model/scalar8kLinear.sav'
            X_train,X_test,y_train,y_test,y_pred,classifier,cm,standardScaler = sv.randomF(X,y,100)
        pickle.dump(classifier, open(FILE_NAME, 'wb'))
        pickle.dump(standardScaler, open(FILE_NAME_SCALAR, 'wb'))
        train(classifier,standardScaler,std = 1)
            
            
    if sys.argv[1] == '-l':
        if sys.argv[2] == 'rbf':
            FILE_NAME = 'Model/model8kRBF.sav'
            FILE_NAME_SCALAR = 'Model/scalar8kRBF.sav'
        if sys.argv[2] == 'rf':
            FILE_NAME = 'Model/model8kRF.sav'
            FILE_NAME_SCALAR = 'Model/scalar8kRF.sav'
        if sys.argv[2] == 'linear':
            FILE_NAME = 'Model/model8kLinear.sav'
            FILE_NAME_SCALAR = 'Model/scalar8kLinear.sav'
            
        classifier = pickle.load(open(FILE_NAME, 'rb'))
        standardScaler = pickle.load(open(FILE_NAME_SCALAR, 'rb'))
        train(classifier,standardScaler,std = 1)

