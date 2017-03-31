import cv2
import numpy as np
src = cv2.imread("testBlock.jpg")


# getHistogramOfGradients(src) get the histogram of gradients of a BGR picture
# Src a opencv imread picture

def getHistogramOfGradients(src):        
    dx = cv2.Sobel(src,cv2.CV_64F,1,0,ksize=1)
    dy = cv2.Sobel(src,cv2.CV_64F,0,1,ksize=1) 
    gradientMagnitude,ang = cv2.cartToPolar(dx,dy,angleInDegrees = True) #[0 ; 360]
    teta = np.rad2deg(np.arctan2(dy,dx)) #OBS:[-180, 180]
    gradientDirection = np.abs(teta) # [0 ; 180]
    
    #Vector
    # 0(180) | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160
    histogramOfGradients = np.zeros(9)
    
    rows,cols,channel = src.shape
    for j in range(0, rows):
        for i in range(0,cols):
            maxIndex  = gradientMagnitude[i][j].argmax() # get Max magnitude (B;G;R)
            hogIndex  = gradientDirection[i][j][maxIndex]/20
            if gradientDirection[i][j][maxIndex] >= 180.0:
                hogIndex = 0
            percentil = 1 - np.mod(gradientDirection[i][j][maxIndex] ,20)/20
            value = gradientMagnitude[i][j][maxIndex] * percentil
            nextValue = gradientMagnitude[i][j][maxIndex] - value
            histogramOfGradients[int(hogIndex)] += value
            histogramOfGradients[int(hogIndex)+1] += nextValue
    return histogramOfGradients

getHistogramOfGradients(src)
                    