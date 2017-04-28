# ObjectDetection
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import sys
from tqdm import tqdm
from random import randint
sys.path.insert(0, 'Example')
import HOG as HOG
import svm as sv
NUM_OF_IMGS_P = 26466  # 13233#4964
NUM_OF_IMGS_N = 4684
# Input will be (86,86,?)
NumberOfDim = 2916


def mergeX(positive=1):
    files2Merge = "list.txt"
    if positive == 1:
        CSV = "CSV/Positive/"
    else:
        CSV = "CSV/Negative/"
    concatList = CSV + files2Merge
    dataList = pd.read_csv(concatList).iloc[:, :].values
    xList = np.empty([1, NumberOfDim], dtype="float32")
    for folder in dataList:
        print "----- Start To Get Folder " + CSV + folder + " -----"
        dataset = pd.read_csv(CSV + folder[0], dtype="float32")
        xNew = dataset.iloc[:, :].values
        xList = np.vstack((xList, xNew))
    xList = np.delete(xList, (0), axis=0)
    return xList


def getX(fromFile=0, positive=1, files=1, idT=0, save=0):
    if positive == 1:
        VAR = "Data/temp/"
        SAVE = "Data/positive4/"
        CSV = "CSV/Positive_SamplesN.csv"
        NUM_OF_IMGS = NUM_OF_IMGS_P
        TYPEFILE = ".jpg"
    else:
        VAR = "Data/negative2/"
        SAVE = "Data/negative3/"
        CSV = "CSV/Negative_Samples4.csv"
        NUM_OF_IMGS = NUM_OF_IMGS_N
        TYPEFILE = ".JPEG"
    if fromFile == 0:
        lista = np.empty([1, 2916])
        for f in tqdm(range(1, NUM_OF_IMGS + 1)):  # 13233
            # Full image set (already cut, in ratio 1:1)
            src = cv2.imread(VAR + str(f) + TYPEFILE)
            rows, cols, channel = src.shape
            if rows > 1 and cols > 1:
                maxRows = rows / 86
                maxCols = cols / 86
                for j in range(0, maxRows):
                    for i in range(0, maxCols):
                        roi, xMin, xMax, yMin, yMax = HOG.getROIsrc(src, i, j, px=86)
                        rowsR, colsR, channel = roi.shape
                        if rowsR != 86 or colsR != 86:
                            roi = cv2.resize(roi, (86, 86))
                        cv2.imwrite(SAVE + str(f) + "j" + str(j) + "i" + str(i) + TYPEFILE, roi)
                        histG = HOG.getHistogramOfGradients(roi)
                        lista = np.vstack((lista, histG))
            else:
                histG = np.zeros(2916)
                lista = np.vstack((lista, histG))
        X = np.delete(lista, (0), axis=0)
        np.savetxt(CSV, X, delimiter=",", fmt="%f")
    else:
        dataset = pd.read_csv(CSV)
        X = dataset.iloc[:, :].values
    return X


def train(classifier, stdScaler, std=0):
    ID = randint(0, 99999)
    if std == 0:
        VAR = "Data/positive2/"
        NUM_OF_IMGS = NUM_OF_IMGS_P
        TYPEFILE = ".ppm"
        for i in tqdm(range(1, NUM_OF_IMGS + 1)):  # 13233
            src = cv2.imread(VAR + str(i) + TYPEFILE)
            src = cv2.resize(src, (86, 86))
            histG = HOG.getHistogramOfGradients(src)
            histGE = stdScaler.transform(histG)
            print classifier.predict(histGE)
    else:
        TYPEFILE = ".jpg"
        DIRECTORY = "Example/test"
        src = cv2.imread(DIRECTORY + TYPEFILE)
        # src = cv2.pyrUp(cv2.pyrDown(src))
        srcUp = src  # cv2.pyrDown(src)
        rows, cols, channel = srcUp.shape
        src2 = srcUp.copy()
        maxRows = rows / 86
        maxCols = cols / 86
        for j in tqdm(range(0, maxRows)):
            for i in range(0, maxCols):
                for dY in range(0, 3):
                    for dX in range(0, 3):
                        roi, xMin, xMax, yMin, yMax = HOG.getROIsrc(srcUp, j, i, px=86, dy=dY*20, dx=dX*20)
                        rows, cols, channel = roi.shape
                        if rows == 0 or cols == 0:
                            break
                        if rows != 86 or cols != 86:
                            roi = cv2.resize(roi, (86, 86))
                        histG = HOG.getHistogramOfGradients(roi)
                        histGE = stdScaler.transform(histG)
                        if classifier.predict(histGE):
                            cv2.rectangle(src2, (yMin, xMin), (yMax, xMax), (0, 0, 255))
                            plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                            cv2.imwrite("Img/"+"ID"+str(ID)+str(j*1000)+str(i*100)+str(dY*10)+str(dX)+"Foi"+".jpg",roi)
        cv2.imwrite("ID" + str(ID) + "Rect.jpg", src2)
        print "The ID: " + str(ID)


def cutPositiveImg():
    VAR = "Data/positive/"
    SAVE = "Data/positive3/"
    NUM_OF_IMGS_TO_CUT = 13233
    TYPEFILE = ".jpg"
    for i in tqdm(range(1, NUM_OF_IMGS_TO_CUT + 1)):
        src = cv2.imread(VAR + str(i) + TYPEFILE)
        rows, cols, channel = src.shape
        if cols == 250 and rows == 250:
            cutImg, xMin, xMax, yMin, yMax = HOG.getROIsrc(
                src, 0, 0, px=180, dx=35, dy=35)
            resizeImg = cv2.resize(cutImg, (86, 86))
            cv2.imwrite(SAVE + str(i) + TYPEFILE, resizeImg)


if len(sys.argv) <= 1:
    print "Error not flags: -x n p || -c rbf rf linear || -l rbf rf linear"
else:
    if sys.argv[1] == '-?':
        print "Error not flags: -x n p || -c rbf rf linear || -l rbf rf linear"
    if sys.argv[1] == '-x':
        if sys.argv[2] == 'n':
            XN = getX(fromFile=0, positive=0)
            rows, cols = XN.shape
            YN = np.zeros(shape=(rows, 1), dtype=int)
        if sys.argv[2] == 'p':
            XP = getX(fromFile=0, positive=1)
            rows, cols = XP.shape
            YP = np.ones(shape=(rows, 1), dtype=int)
    if sys.argv[1] == '-c':
        print "------Getting Negative Samples from File------"
        XN = mergeX(positive=0)
        rowsN, colsN = XN.shape
        YN = np.zeros(shape=(rowsN, 1), dtype=int)
        print "------Getting Positive Samples from File------"
        XP = mergeX(positive=1)
        rowsP, colsP = XP.shape
        YP = np.ones(shape=(rowsP, 1), dtype=int)
        X = np.vstack((XP, XN))
        y = np.vstack((YP, YN))
        y = y.ravel()

        print "---------------Start The Model----------------"
        if sys.argv[2] == 'rbf':
            FILE_NAME = 'Model/model8kRBF.sav'
            FILE_NAME_SCALAR = 'Model/scalar8kRBF.sav'
            X_train, X_test, y_train, y_test, y_pred, classifier, cm, standardScaler = sv.svm(X, y, 'rbf')
        if sys.argv[2] == 'rf':
            FILE_NAME = 'Model/modelTkRF.sav'
            FILE_NAME_SCALAR = 'Model/scalarTkRF.sav'
            X_train,X_test,y_train,y_test,y_pred,classifier,cm,standardScaler = sv.randomF(X, y, 100)
        if sys.argv[2] == 'linear':
            FILE_NAME = 'Model/modelALLLinear.sav'
            FILE_NAME_SCALAR = 'Model/scalarALLLinear.sav'
            X_train,X_test,y_train,y_test,y_pred,classifier,cm,standardScaler = sv.svmLinear(X, y, 0.01)

        print "-----------------Save The Model---------------"
        pickle.dump(classifier, open(FILE_NAME, 'wb'))
        pickle.dump(standardScaler, open(FILE_NAME_SCALAR, 'wb'))

        print "-----------------Train The Model--------------"
        train(classifier, standardScaler, std=1)

    if sys.argv[1] == '-l':
        if sys.argv[2] == 'rbf':
            FILE_NAME = 'Model/model8kRBF.sav'
            FILE_NAME_SCALAR = 'Model/scalar8kRBF.sav'
        if sys.argv[2] == 'rf':
            FILE_NAME = 'Model/modelTkRF.sav'
            FILE_NAME_SCALAR = 'Model/scalarTkRF.sav'
        if sys.argv[2] == 'linear':
            FILE_NAME = 'Model/modelALLLinear.sav'
            FILE_NAME_SCALAR = 'Model/scalarALLLinear.sav'
        classifier = pickle.load(open(FILE_NAME, 'rb'))
        standardScaler = pickle.load(open(FILE_NAME_SCALAR, 'rb'))
        train(classifier, standardScaler, std=1)
