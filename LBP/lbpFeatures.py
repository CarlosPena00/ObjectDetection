# LBP Feature extractor with R=1 fix
import cv2 as cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# Open image
image = cv2.imread("../Example/test.jpg")
# Convert image bgr to gray
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# get rows and cols len of the image
rows = image.shape[0]
cols = image.shape[1]

#cv2.imshow("image",image)
#cv2.waitKey(0)

# initialize empty list of lbp features
myList = []
totals = 0
for i in range(1,rows-1):
	for j in range(1,cols-1):
		# Boards are not treated
		#if(~(i == rows-1 or i == 1 or j == cols-1 or j == 0)):
			# Get central pixel to calculate lbp features
		pc = image[i,j]

		# Calculate threshold of the central pixel and others in 3x3 window
		# if px >= pc : px = 1, else px = 0, making the binary code
		p0 = 1 if image[i-1,j-1] >= pc else 0
		p1 = 1 if image[i-1,j] >= pc else 0
		p2 = 1 if image[i-1,j+1] >= pc else 0
		p3 = 1 if image[i,j+1] >= pc else 0
		p4 = 1 if image[i+1,j+1] >= pc else 0
		p5 = 1 if image[i+1,j] >= pc else 0
		p6 = 1 if image[i+1,j-1] >= pc else 0
		p7 = 1 if image[i,j-1] >= pc else 0

		# Convert binary code to decimal representation
		p0 = math.pow(2,0)*p0
		p1 = math.pow(2,1)*p1
		p2 = math.pow(2,2)*p2
		p3 = math.pow(2,3)*p3
		p4 = math.pow(2,4)*p4
		p5 = math.pow(2,5)*p5
		p6 = math.pow(2,6)*p6
		p7 = math.pow(2,7)*p7
		summ = p0+p1+p2+p3+p4+p5+p6+p7
		summ = int(summ)
		totals = summ
		#print [summ,p0,p1,p2,p3,p4,p5,p6,p7]
		myList.append(summ)

# Plot Lbp data
 
plt.hist(myList, bins=np.arange(0,255), align='left', normed='True')
plt.xlim(0,255)
plt.show()