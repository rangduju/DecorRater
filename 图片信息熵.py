# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:16:31 2020

@author: Micalry Xavier
"""
import math
import cv2
import numpy as np
tmp = []
for i in range(256):
    tmp.append(0)
val = 0
k = 0
res = 0
image=cv2.imread('D:/opencvpic/1.JPG',0)
#image = cv2.imread('C:/Users/shaw/Desktop/new/result/stand/squat_0.5_5_1.png',0)
img = np.array(image)
def entropy(img):
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k =  float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res
