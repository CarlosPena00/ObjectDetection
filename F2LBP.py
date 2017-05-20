#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:33:02 2017

@author: Carlos Pena
@author: Heitor Rapela
"""

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'Source')
import HOG as HOG
sys.path.insert(0, 'LBP')
import lbpFeatures as LBP
import time
IMSIZE = 76

def folds2LBP(fromFile=0, positive=1, argMin=0, argMax=100, Blur = 0):
    DATAFOLDER = "Data/"
    CSVFOLDER = "CSV/"
    if positive == 0 and fromFile == 0:
        CSVFOLDER += "Negative/"
        data = pd.read_csv("Data/negativeList.csv")
    if positive == 1 and fromFile == 0:
        CSVFOLDER += "Positive/"
        data = pd.read_csv("Data/positiveList.csv")

    foldName = data.iloc[:, :].values
    minIndex = max(argMin, 0)
    maxIndex = min(foldName.shape[0], argMax)
    for i in range(minIndex, maxIndex):
        FOLDER = foldName[i][0]
        TYPEFILE = "." + foldName[i][1]
        CUT = foldName[i][2]
        NUMBER_OF_IMG = foldName[i][3]
        SAVEFILE = FOLDER[:-1] + ".csv"
        print "----- Start Folder: " + FOLDER + " -----"
        time.sleep(1)
        fold2LBP(DATAFOLDER, CSVFOLDER, FOLDER,
                 TYPEFILE, CUT, NUMBER_OF_IMG, SAVEFILE, Blur)


def fold2LBP(DATAFOLDER, CSVFOLDER, FOLDER, TYPEFILE, CUT, NUMBER_OF_IMG, SAVEFILE, Blur):
    lista = np.empty([1,5477]) #np.empty([1, 2369])
    for f in tqdm(range(1, NUMBER_OF_IMG + 1)):
        src = cv2.imread(DATAFOLDER + FOLDER + str(f) + TYPEFILE)
        rows, cols, channel = src.shape
            
        if rows > 1 and cols > 1:
        
            if Blur == 1:
                src = cv2.pyrUp(cv2.pyrDown(src))
            rows, cols, channel = src.shape
            
            if CUT == 1:
                maxRows = rows / IMSIZE
                maxCols = cols / IMSIZE
            if CUT == 0:
                maxRows = 1
                maxCols = 1
            for j in range(0, maxRows):
                for i in range(0, maxCols):
                    if CUT == 1:
                        roi, xMin, xMax, yMin, yMax = HOG.getROIsrc(
                            src, j, i, px=IMSIZE)
                    if CUT == 0:
                        roi = src
                    rowsR, colsR, channel = roi.shape
                    
                    if rowsR < 1 or colsR < 1:
                        print "F2H Fold2LBP erro Size"
                        continue
                    
                    if rowsR != IMSIZE or colsR != IMSIZE:
                        roi = cv2.resize(roi, (IMSIZE, IMSIZE))
                    histLBP = LBP.getHistogramOfLBP(roi)
                    # print lista.shape , histLBP.shape, roi.shape
                    lista = np.vstack((lista, histLBP))
        else:
            print "Error Img not fond"
            histLBP = np.zeros(5477)
            lista = np.vstack((lista, histLBP))

    X = np.delete(lista, (0), axis=0)
    np.savetxt("LBP_"+CSVFOLDER + SAVEFILE, X, delimiter=",", fmt="%f")


if __name__ == "__main__":
    print "Positive min max Blur"
    if len(sys.argv) > 4:
        pos = int(sys.argv[1])
        minV = int(sys.argv[2])
        maxV = int(sys.argv[3])
        Blur = int(sys.argv[4])
        
        print minV, maxV
    else:
        print "Argv no info, using default"
        pos = 0
        minV = 5
        maxV = 6
        Blur = 0
    folds2LBP(positive=pos, argMin=minV, argMax=maxV + 1, Blur=Blur)