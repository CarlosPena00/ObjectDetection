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

FILE_NAME = 'finalized_model8k.sav'
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
        if sys.argv[2] == 'C':
            XN = getX(fromFile = 1, positive= 0)
            YN = np.zeros(shape=(4674,1), dtype = int)
            XP = getX(fromFile = 1, positive= 1)
            YP = np.ones(shape=(4963,1),dtype = int)
            X = np.vstack((XP,XN))
            y = np.vstack((YP,YN))
            X_train,X_test,y_train,y_test,y_pred,classifier,cm = sv.svm(X,y)
            pickle.dump(classifier, open(FILE_NAME, 'wb'))
            print cm
    if sys.argv[1] == '-l':
        if sys.argv[2] == 'C':
            loaded_model = pickle.load(open(FILE_NAME, 'rb'))


