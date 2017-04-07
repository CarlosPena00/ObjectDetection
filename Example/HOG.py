import cv2
import numpy as np
import matplotlib.pyplot as plt


def getROI(gMag,gDir, idY, idX = 0 ,px = 6 ):
    rMag = gMag[(px+(idX-1)*px):(px+idX*px) ,  (px+(idY-1)*px):(px+idY*px), :]
    rDir = gDir[(px+(idX-1)*px):(px+idX*px) ,  (px+(idY-1)*px):(px+idY*px), :]
    return rMag,rDir


def cart2Polar(src):
    dx = cv2.Sobel(src,cv2.CV_64F,1,0,ksize=1)
    dy = cv2.Sobel(src,cv2.CV_64F,0,1,ksize=1) 
    gradientMagnitude,ang = cv2.cartToPolar(dx,dy,angleInDegrees = True) #[0 ; 360]
    teta = np.rad2deg(np.arctan2(dy,dx)) #OBS:[-180, 180]
    gradientDirection = np.abs(teta) # [0 ; 180]
    return gradientMagnitude,gradientDirection


def getSimpleHOG(rMag,rDir):
    histogramOfGradients = np.zeros(9)
    cols,rows,channel = rMag.shape
    for j in range(0, rows):
        for i in range(0,cols):
            maxIndex  = rMag[i][j].argmax() # get Max magnitude (B;G;R)
            hogIndex  = rDir[i][j][maxIndex]/20
            if rDir[i][j][maxIndex] >= 180.0:
                hogIndex = 0
            percentil = 1 - np.mod(rDir[i][j][maxIndex] ,20)/20
            value = rMag[i][j][maxIndex] * percentil
            nextValue = rMag[i][j][maxIndex] - value
            histogramOfGradients[int(hogIndex)] += value
            nextIndex = int(hogIndex)+1
            if nextIndex >= 9:
                nextIndex = 0
            histogramOfGradients[nextIndex] += nextValue
    return histogramOfGradients

def getHistogramOfGradients(src):        
    gMag, gDir = cart2Polar(src) 
    #Vector
    # 0(180) | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160
    cols,rows,channel = src.shape
    fullHOG = []
    #Get Block 3x3 cells
    for delX in range(0,(cols/6)-2):
        for delY in range (0,(rows/6)-2):
            
            #Block Area -> 3x3 cells
            rMag, rDir = getROI(gMag,gDir,delY,delX,px = 18)#6*3
            
            for i in range (0,3):
                for j in range(0,3):        
                    cMag,cDir = getROI(gMag,gDir,i,j)                    
                    #cellHOG.append(getSimpleHOG(rMag,rDir))
                    fullHOG = np.hstack((fullHOG,getSimpleHOG(cMag,cDir)))
    return fullHOG

#src = cv2.imread("test.jpg")
#a = getHistogramOfGradients(src)
