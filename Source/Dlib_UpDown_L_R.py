#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 21:21:17 2017

@author: kaka
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

PDIR = "Data/dlib_Pos/"

DATAFOLDER = "Data/"
CSVFOLDER = "CSV/"
CSVFOLDER += "Positive/"
data = pd.read_csv("Data/positiveList.csv")

foldName = data.iloc[:, :].values

FOLDER = foldName[6][0]
TYPEFILE = "." + foldName[6][1]
CUT = foldName[6][2]
NUMBER_OF_IMG = foldName[6][3]

SAVEFILE = FOLDER[:-1] + ".csv"

for i in tqdm(range(1, NUMBER_OF_IMG+1)):
    src = cv2.imread(DATAFOLDER+FOLDER+str(i)+TYPEFILE)
    #srcDown = cv2.pyrDown(src)
    #srcUpDown = cv2.pyrUp(srcDown)
    srcResize = cv2.resize(src,(86,86))
    
    srcFlip =cv2.flip(srcResize,1)
    cv2.imwrite(DATAFOLDER+"dlib_Pos/Normal_jpg/"+str(i)+TYPEFILE,srcResize)
    cv2.imwrite(DATAFOLDER+"dlib_Pos/Normal_jpg/"+str(i)+"F"+TYPEFILE,srcFlip)


