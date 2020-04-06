# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 11:19:16 2020

@author: Micalry Xavier
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread('D:\opencvpic\1.PNG')
source=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#均值滤波
result=cv2.blur(source,(5,5))

#显示图形
titles=['Source Image','Blue Image']
images=[source,result]
for i in range(2):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
 
    
plt.show()   

