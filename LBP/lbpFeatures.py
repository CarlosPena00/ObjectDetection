# LBP Feature extractor with R=1 fix
import cv2 as cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time

# Open image
image = cv2.imread("teste.jpg")
# Convert image bgr to gray
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# get rows and cols len of the image
rows = image.shape[0]
cols = image.shape[1]

#cv2.imshow("image",image)
#cv2.waitKey(0)

# initialize empty list of lbp features
myList = []
for i in range(1,rows-1):
	for j in range(1,cols-1):
		# Boards are not treated
		#if(~(i == rows-1 or i == 1 or j == cols-1 or j == 0)):
			# Get central pixel to calculate lbp features
		pc = image[i,j]

		p0 = 1 if image[i-1,j-1] >= pc else 0
		p1 = 1 if image[i-1,j] >= pc else 0
		p2 = 1 if image[i-1,j+1] >= pc else 0
		p3 = 1 if image[i,j+1] >= pc else 0
		p4 = 1 if image[i+1,j+1] >= pc else 0
		p5 = 1 if image[i+1,j] >= pc else 0
		p6 = 1 if image[i+1,j-1] >= pc else 0
		p7 = 1 if image[i,j-1] >= pc else 0

		p0 = (int(p0))	
		p1 = (int(p1)* (2 << 0))
		p2 = (int(p2)* (2 << 1))
		p3 = (int(p3)* (2 << 2))
		p4 = (int(p4)* (2 << 3))
		p5 = (int(p5)* (2 << 4))
		p6 = (int(p6)* (2 << 5))
		p7 = (int(p7)* (2 << 6))
		summ = p0+p1+p2+p3+p4+p5+p6+p7
		summ = int(summ)
		myList.append(summ)

# Plot Lbp data
plt.hist(myList, bins=np.arange(0,255), align='left', normed='True')
plt.xlim(0,255)
plt.show()