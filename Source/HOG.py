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
    xMin = (px + (idX - 1) * px)
    xMax = xMin + px
    yMin = (px + (idY - 1) * px)
    yMax = yMin + px
    
    rMag = gMag[xMin:xMax, yMin:yMax, :]
    rDir = gDir[xMin:xMax, yMin:yMax, :]
    return rMag, rDir 
# getROIsrc get region of interest, part of matrix
# src input Matrix
# idY index of the block/cell
# idX index of the block/cell
# px shape of the block/cell (px,px,;)


def getROIsrc(src, idY, idX, px=8, dy=0, dx=0):
    xMin = (dx + px + (idX - 1) * px)
    xMax = xMin + px
    yMin = (dy + px + (idY - 1) * px)
    yMax = yMin + px
    return src[xMin:xMax, yMin:yMax, :], xMin, xMax, yMin, yMax


def getBlock(src,idY,idX,px=16):
    xMin = idX * (px/2)
    xMax = xMin + px
    yMin = idY * (px/2)
    yMax = yMin + px
    return src[xMin:xMax, yMin:yMax, :]


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


def getSimpleHOG(rMag, rDir):
    histogramOfGradients = np.zeros(9,dtype="float32")
    cols, rows, channel = rMag.shape
    for j in range(0, rows):
        for i in range(0, cols):
            maxIndex = rMag[i][j].argmax()  # get Max magnitude (B;G;R)
            hogIndex = rDir[i][j][maxIndex] / 20
            if rDir[i][j][maxIndex] >= 180.0:
                hogIndex = 0
            percentil = 1 - np.mod(rDir[i][j][maxIndex], 20) / 20
            value = rMag[i][j][maxIndex] * percentil
            nextValue = rMag[i][j][maxIndex] - value
            histogramOfGradients[int(hogIndex)] += value
            nextIndex = int(hogIndex) + 1
            if nextIndex >= 9:
                nextIndex = 0
            histogramOfGradients[nextIndex] += nextValue
    return histogramOfGradients


def getSimpleHOGMap(rMag, rDir):
    histogramOfGradients = np.zeros(9,dtype="float32")
    histogramOfGradients1 = np.zeros(9,dtype="float32")
    histogramOfGradients2 = np.zeros(9,dtype="float32")
    rMag = np.asarray(rMag)
    R, G, B = np.float32(rMag[:, :, 0]), np.float32(rMag[:, :, 1]), np.float32(rMag[:, :, 2])
    Max = np.argmax([R.flatten(), G.flatten(), B.flatten()], axis=0)
    Max.reshape(R.shape)
    aux1 = Max == 0
    aux2 = Max == 1
    aux3 = Max == 2
    mask = np.zeros(rMag.shape, dtype=bool)
    mask[:, :, 0] = aux1.reshape(mask[:, :, 0].shape)
    mask[:, :, 1] = aux2.reshape(mask[:, :, 1].shape)
    mask[:, :, 2] = aux3.reshape(mask[:, :, 2].shape)
    maskedRdir = np.float32(rDir[mask].reshape(Max.shape))
    maskedRmag = np.float32(rMag[mask].reshape(Max.shape))
    hogIndex = maskedRdir / 20
    hogIndex[hogIndex >= 9] = 0
    percentil = np.ones(Max.shape,dtype="float32") - np.mod(maskedRdir, 20) / 20
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


# getHistogramOfGradients return the full histogram concat (1,X)
# src input matrix
def getOpenCVHOG(image):
     
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize,
                            blockStride, cellSize, nbins,
                            derivAperture, winSigma,
                            histogramNormType, L2HysThreshold,
                            gammaCorrection, nlevels)
    # compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8, 8)
    padding = (8, 8)
    locations = ((10, 20), )
    hist = hog.compute(image, winStride, padding, locations)
    return hist



def getHistogramOfGradients(src):
    gMag, gDir = cart2Polar(src)
    # Vector
    # 0(180) | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160
    cols, rows, channel = src.shape
    fullHOG = np.float32()
    # Get Block 3x3 cells
    maxX = (cols / 8) - 1
    maxY = (rows / 8) - 1

    for delX in range(0, maxX):
        for delY in range(0, maxY):
            cellHOG = np.float32()
            rMag = getBlock(gMag,delY,delX,px=16)
            rDir = getBlock(gDir,delY,delX,px=16)
            #cv2.imwrite("test/"+str(delX)+str(delY)+".jpg",mimSrc[0])
            for i in range(0, 2):  # 2x2 Block
                for j in range(0, 2):
                    cMag, cDir = getROI(rMag, rDir, i, j, px=8)
                    cellHOG = np.hstack((cellHOG, getSimpleHOGMap(cMag, cDir)))
                    
            summatory = np.sum(cellHOG) + 0.1
            cellHOG = np.sqrt(cellHOG / summatory)
            fullHOG = np.append(fullHOG, cellHOG)
    fullHOGMatrix = np.asmatrix(fullHOG)
    return fullHOGMatrix


if __name__ == "__main__":
    dim = 76

    src = cv2.imread("../TestImg/test8.jpg")
    src[6, 6] = (0,0,0)
    src[5, 5] = (0,0,0)
    src[5, 6] = (0,0,0)
    src[6, 5] = (0,0,0)
    src = cv2.resize(src,(dim, dim))

# src2 = cv2.resize(src,(64,128))
# print src2.shape
# hog = getHistogramOfGradients(src3)
# src = cv2.pyrDown(src)
# src = cv2.pyrDown(src)
# a = getHistogramOfGradients(src)
# b = getHistogramOfGradients(src)
# c = np.vstack((a,b))
