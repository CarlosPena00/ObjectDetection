import cv2
import numpy as np
src = cv2.imread("testBlock.jpg")
dx = cv2.Sobel(src,cv2.CV_64F,1,0,ksize=1)
dy = cv2.Sobel(src,cv2.CV_64F,0,1,ksize=1) 
mag,ang = cv2.cartToPolar(dx,dy,angleInDegrees = True) #[0 ; 360]
teta = np.rad2deg(np.arctan2(dy,dx)) #OBS:[-180, 180]
teta = np.abs(teta) # [0 ; 180]

#Vector
# 0(180) | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160
np.zeros(9)
rows,cols,channel = src.shape

index = np.zeros(3)
for j in range(0, rows):
    for i in range(0,cols):
        maxIndex = mag[i][j].argmax() # get Max magnitude (B;G;R)
        
            