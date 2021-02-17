# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 23:23:23 2021

@author: yash
"""

import cv2
minValue = 70
def processing(path):    
    frame = path
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)

    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    
    return th3

