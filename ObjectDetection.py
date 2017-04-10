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

FILE_NAME = 'finalized_model20k.sav'
#Input will be (125,125,?)
def getX(fromFile = 0,positive = 1):
    if positive == 1:
        VAR = "Data/positive/"
        CSV = "CSV/Positive_Samples.csv"
        NUM_OF_IMGS =13233
        TYPEFILE = ".jpg"
    else:
        VAR = "Data/negative/"
        CSV = "CSV/Negative_Samples.csv"
        NUM_OF_IMGS = 7512
        TYPEFILE = ".JPEG"
    if fromFile == 0:        
        lista = np.empty([1,7056])
        for i in tqdm(range(1,NUM_OF_IMGS+1)):#13233
            #Full image set (already cut, in ratio 1:1)
            src = cv2.imread(VAR+str(i)+TYPEFILE)
            src = cv2.resize(src,(125,125))
            histG = HOG.getHistogramOfGradients(src)            
            lista = np.vstack((lista,histG))
            #lista.extend(histG)        
           
        X = np.delete(lista,(0),axis=0)
        np.savetxt(CSV,X,delimiter= ",",fmt="%.2f")
    else:
        dataset = pd.read_csv(CSV)
        X = dataset.iloc[:,:].values
    return X

XN = getX(fromFile = 1, positive= 0)
XP = getX(fromFile = 1, positive= 1)

YP = np.ones(shape=(13232,1),dtype = int)
YN = np.zeros(shape=(7511,1), dtype = int)

X = np.vstack((XP,XN))
y = np.vstack((YP,YN))

Matrix = np.hstack((X,y))
fromFile = 1
if fromFile == 0:
    X_train,X_test,y_train,y_test,y_pred,classifier,cm = sv.svm(X,y)
    pickle.dump(classifier, open(FILE_NAME, 'wb'))
else:
    loaded_model = pickle.load(open(FILE_NAME, 'rb'))
#result = loaded_model.score(X_test, y_test)

#print(result)

