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
sys.path.insert(0, 'Source')
import HOG as HOG
import machineLearning as ml
NUM_OF_IMGS_P = 26466  # 13233#4964
NUM_OF_IMGS_N = 4684
# Input will be (86,86,?)
NumberOfDim = 2369
IMSIZE = 76

# Malisiewicz et al. (overlapThresh  are normally between 0.3 and 0.5.)
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        print "Boxes Vazio"
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes 
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")



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
            src = cv2.resize(src, (IMSIZE, IMSIZE))
            histG = HOG.getHistogramOfGradients(src)
            histGE = stdScaler.transform(histG)
            print classifier.predict(histGE)
    else:
        TYPEFILE = ".jpg"
        DIRECTORY = "TestImg/test9"
        src = cv2.imread(DIRECTORY + TYPEFILE)
        #src = cv2.pyrUp(src)
        srcUp = src #cv2.pyrDown(src)

        # srcUp = cv2.pyrUp( cv2.pyrDown(src))

        rows, cols, channel = srcUp.shape
        src2 = srcUp.copy()
        maxRows = rows / IMSIZE
        maxCols = cols / IMSIZE
        rects = []
        for j in tqdm(range(0, maxRows)):
            for i in range(0, maxCols):
                for dY in range(0, 3):
                    for dX in range(0, 3):
                        roi, xMin, xMax, yMin, yMax = HOG.getROIsrc(srcUp, j, i, px=IMSIZE, dy=dY*15, dx=dX*15)
                        rows, cols, channel = roi.shape
                        if rows == 0 or cols == 0:
                            break
                        if rows != IMSIZE or cols != IMSIZE:
                            roi = cv2.resize(roi, (IMSIZE, IMSIZE))
                        histG = HOG.getHistogramOfGradients(roi)
                        histGE = stdScaler.transform(histG)
                        
                        if classifier.predict(histGE):
                            #cv2.rectangle(src2, (yMin, xMin), (yMax, xMax), (0, 0, 255))
                            recs_aux = np.array([xMin, yMin, xMax, yMax]) 
                            rects.append(recs_aux)
                            plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                            cv2.imwrite("Img/"+"ID"+str(ID)+str(j*1000)+str(i*100)+str(dY*10)+str(dX)+"Foi"+".jpg",roi)

        
        boxes = non_max_suppression_fast(np.asarray(rects), 0.3)

        for bx in boxes:
            xMin = bx[0]
            yMin = bx[1]
            xMax = bx[2]
            yMax = bx[3]
            cv2.rectangle(src2, (yMin, xMin), (yMax, xMax), (0, 255, 0))
            #print "Box detectado"
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
            X_train, X_test, y_train, y_test, y_pred, classifier, cm, standardScaler = ml.svm(X, y, 'rbf')
        if sys.argv[2] == 'rf':
            FILE_NAME = 'Model/modelTkRF.sav'
            FILE_NAME_SCALAR = 'Model/scalarTkRF.sav'
            X_train,X_test,y_train,y_test,y_pred,classifier,cm,standardScaler = ml.randomF(X, y, 100)
        if sys.argv[2] == 'linear':
            FILE_NAME = 'Model/modelLinear.sav'
            FILE_NAME_SCALAR = 'Model/scalarLinear.sav'
            X_train,X_test,y_train,y_test,y_pred,classifier,cm,standardScaler = ml.svmLinear(X, y, 0.01)

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
            FILE_NAME = 'Model/modelLinear.sav'
            FILE_NAME_SCALAR = 'Model/scalarLinear.sav'
        classifier = pickle.load(open(FILE_NAME, 'rb'))
        standardScaler = pickle.load(open(FILE_NAME_SCALAR, 'rb'))
        train(classifier, standardScaler, std=1)
