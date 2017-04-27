#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 22:26:53 2017

@author: kaka
"""
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
DIR = "Data/dlib_face_detection_dataset/"
NDIR = "Data/dlib_Neg/"
PDIR = "Data/dlib_Pos/"
PIDIR = "Data/dlib_Pos_Ignore/"
DIRXML = DIR+"faces_2016_09_30.xml"
tree = ET.parse(DIRXML)
root = tree.getroot()


images  = root.find("images")

negCount = 0
posCount = 0
posWin = 0
igCount = 0
for image in tqdm(images.iter("image")):

    
    imgDir = image.get("file")
    TYPE = imgDir[imgDir.rfind('.'):]
    imgFullDir = DIR + imgDir
    src = cv2.imread(imgFullDir)
    img = src
    
    if image.find("box") == None:
        negCount += 1
        cv2.imwrite(NDIR + str(negCount) + TYPE,img)
    else:
        posCount += 1
        for box in image.findall("box"):
            top = int(box.attrib.get("top"))
            left = int(box.attrib.get("left"))
            width = int(box.attrib.get("width"))
            height = int(box.attrib.get("height"))
            ignore = box.attrib.get("ignore")
            if left < 0 :
                left = 0
            if top < 0 :
                top = 0
            if width < 0 :
                width = 0
            if height < 0 :
                height = 0
                
            if ignore is not None:
                igCount += 1
                img = src[top:top+height,left:left + width,:]
                cv2.imwrite(PIDIR + str(igCount) + TYPE,img)
            else:
                posWin += 1
                img = src[top:top+height,left:left + width,:]
                cv2.imwrite(PDIR + str(posWin) + TYPE,img)
    #plt.imshow(img)
    #plt.plot
print posCount, posWin, igCount, negCount


# 7213it [00:21, 337.60it/s]6331 11480 882


