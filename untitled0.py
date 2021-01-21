# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 23:17:35 2021

@author: hp
"""
from image_preprocessing import processing 
import os
import cv2
minValue=70
frame = cv2.imread('1.jpeg')
    
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),2)
th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("1st",res)
cv2.waitKey(0)
cv2.destroyAllWindows()
