#ObjectDetection
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Example')
import cart2PolarOpencv as HOG

src = cv2.imread("Data/src/1.jpg")
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
src = cv2.pyrDown(src)
src = cv2.pyrDown(src)
src = cv2.pyrDown(src)
src = cv2.pyrDown(src)
plt.imshow(src)

histG = HOG.getHistogramOfGradients(src)
print histG