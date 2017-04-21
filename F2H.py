#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:33:02 2017

@author: Carlos Pena 
"""

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, 'Example')
import HOG as HOG
import time

def folds2Hog(fromFile = 0,positive = 1, argMin = 0, argMax = 100):
    DATAFOLDER = "Data/"
    CSVFOLDER = "CSV/"
    if positive == 0 and fromFile == 0:
        CSVFOLDER += "Negative/"
        data = pd.read_csv("Data/negativeList.csv")
        foldName = data.iloc[:,:].values
        minIndex = max(argMin,0)
        maxIndex = min(foldName.shape[0],argMax)
        for i in range(minIndex,maxIndex):
            FOLDER = foldName[i][0]      
            TYPEFILE = "."+foldName[i][1]
            CUT = foldName[i][2]
            NUMBER_OF_IMG = foldName[i][3]
            SAVEFILE = FOLDER[:-1]+".csv"
            print "----- Start Folder: "+FOLDER+" -----"
            time.sleep(1)
            fold2Hog(DATAFOLDER,CSVFOLDER,FOLDER,TYPEFILE,CUT,NUMBER_OF_IMG,SAVEFILE,)
	        
def fold2Hog(DATAFOLDER,CSVFOLDER,FOLDER,TYPEFILE,CUT,NUMBER_OF_IMG,SAVEFILE):
    lista = np.empty([1,2916])
    for f in tqdm(range(1,NUMBER_OF_IMG+1)):
        src = cv2.imread(DATAFOLDER+FOLDER+str(f)+TYPEFILE)
        rows,cols,channel = src.shape 
        if rows > 1 and cols > 1:
            maxRows = rows/86
            maxCols = cols/86
            for j in range(0,maxRows):
                for i in range(0, maxCols):
                    roi,xMin,xMax,yMin,yMax = HOG.getROIsrc(src,i,j,px = 86)
                    rowsR,colsR,channel = roi.shape
                    if rowsR != 86 or colsR != 86:
                        roi = cv2.resize(roi,(86,86))
                    histG = HOG.getHistogramOfGradients(roi)
                    lista = np.vstack((lista,histG))
        else:
            histG = np.zeros(2916)
            lista = np.vstack((lista,histG))     
       
    X = np.delete(lista,(0),axis=0)
    np.savetxt(CSVFOLDER+SAVEFILE,X,delimiter= ",",fmt="%f")

if __name__ == "__main__":
	minV = int(sys.argv[1])
	maxV = int(sys.argv[2])
	print minV, maxV
	folds2Hog(positive=0,argMin=minV, argMax=maxV)