## @package HOG
 
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# getROI get region of interest, part of matrix
# gMag gradient magnitude
# gDir gradient direction
# idY index of the block/cell
# idX index of the block/cell
# px shape of the block/cell (px,px,;)
 
 
def getROI(gMag, gDir, idY, idX=0, px=8):
    rMag = gMag[(px + (idX - 1) * px):(px + idX * px) ,  (px + (idY - 1) * px):(px + idY * px), :]
    rDir = gDir[(px + (idX - 1) * px):(px + idX * px), (px + (idY - 1) * px):(px + idY * px), :]
    return rMag, rDir
 
# getROIsrc get region of interest, part of matrix
# src input Matrix
# idY index of the block/cell
# idX index of the block/cell
# px shape of the block/cell (px,px,;)
 
 
def getROIsrc(src, idY, idX, px=8, dy=0, dx=0):
    xMin = (dx + px + (idX - 1) * px)
    xMax = (dx + px + idX * px)
    yMin = (dy + px + (idY - 1) * px)
    yMax = (dy + px + idY * px)
    return src[xMin:xMax, yMin:yMax, :], xMin, xMax, yMin, yMax
 
# cart2Polar return gradientMagnitude,gradientDirection of a img
# src input matrix
 
 
def cart2Polar(src):
    dx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=1)
    dy = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=1)
    gM, ang = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    teta = np.rad2deg(np.arctan2(dy, dx))  # OBS:[-180, 180]
    gradientDirection = np.abs(teta)  # [0 ; 180]
    return gM, gradientDirection
 
# getSimpleHOG return histogram of Gradients of simgle cell
# rMag Mag of the cell
# rDir Dir of the cell
# return a  vector of gradients 9 bins
 
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

def getSimpleHogMap(rMag, rDir):
    histogramOfGradients = np.zeros(9)
    histogramOfGradients1 = np.zeros(9)
    histogramOfGradients2 = np.zeros(9)
    rMag = np.asarray(rMag)
    R, G, B = rMag[:, :, 0], rMag[:, :, 1], rMag[:, :, 2]
    Max = np.argmax([R.flatten(), G.flatten(), B.flatten()], axis=0)
    Max.reshape(R.shape)
    aux1 = Max == 0
    aux2 = Max == 1
    aux3 = Max == 2
    mask = np.zeros(rMag.shape, dtype=bool)
    mask[:, :, 0] = aux1.reshape(mask[:, :, 0].shape)
    mask[:, :, 1] = aux2.reshape(mask[:, :, 1].shape)
    mask[:, :, 2] = aux3.reshape(mask[:, :, 2].shape)
    maskedRdir = rDir[mask].reshape(Max.shape)
    maskedRmag = rMag[mask].reshape(Max.shape)
    hogIndex = maskedRdir / 20
    hogIndex[hogIndex >= 9] = 0
    percentil = np.ones(Max.shape) - np.mod(maskedRdir, 20) / 20
    value = maskedRmag * percentil
    nextValue = maskedRmag - value
    b = np.floor(hogIndex.flatten())
    c = value.flatten()
    N = b.shape[0]
    M = 9
    histogramOfGradients1 = (((np.mgrid[:M, :N]) == b)[0] * c).sum(axis=1)
    # histogramOfGradients[hogIndex.flatten()] += value.flatten()
    nextIndex = np.floor(hogIndex) + np.ones(hogIndex.shape)
    nextIndex[nextIndex > 8] = 0
    nextIndex[nextIndex < 0] = 0
    b = nextIndex.flatten()
    c = nextValue.flatten()
    N = b.shape[0]
    histogramOfGradients2 = (((np.mgrid[:M, :N]) == b)[0] * c).sum(axis=1)
    histogramOfGradients = histogramOfGradients1 + histogramOfGradients2
    # histogramOfGradients[nextIndex] += nextValue
    
    return histogramOfGradients





##getHistogramOfGradients return the full histogram concat (1,X)
#src input matrix
def getHistogramOfGradients(src):        
    gMag, gDir = cart2Polar(src) 
    #Vector
    # 0(180) | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160
    cols,rows,channel = src.shape
    fullHOG = []
    #Get Block 3x3 cells
    for delX in range(0,(cols/8)-1):
        for delY in range (0,(rows/8)-1):
            cellHOG = []
            #Block Area -> 3x3 cells
            rMag, rDir = getROI(gMag,gDir,delY,delX,px = 16)#6*3 or 8*2
            
            for i in range (0,2): # 2x2 Block
                for j in range(0,2):        
                    cMag,cDir = getROI(gMag,gDir,i,j)                    
                    #cellHOG.append(getSimpleHOG(rMag,rDir))
                    cellHOG = np.hstack((cellHOG,getSimpleHOG(cMag,cDir)))
            summatory = np.sum(cellHOG)+0.1
            cellHOG = np.sqrt( cellHOG /summatory)
            fullHOG = np.append(fullHOG,cellHOG)
    fullHOGMatrix = np.asmatrix(fullHOG)
    return fullHOGMatrix


##Just for debug
#src = cv2.imread("Data/positive/1.jpg")
#print src.shape
#src2 = cv2.resize(src,(64,128))
#print src2.shape
#hog = getHistogramOfGradients(src3)
#src = cv2.pyrDown(src)
#src = cv2.pyrDown(src)
#a = getHistogramOfGradients(src)
#b = getHistogramOfGradients(src)
#c = np.vstack((a,b))