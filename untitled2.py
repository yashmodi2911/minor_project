# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 23:05:01 2021

@author: hp
"""

import cv2
import os
import numpy as np
from keras.models import model_from_json
from image_preprocessing import processing
path=os.getcwd() + "\\check"

json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

while True:
    __,frame=cap.read()
    frame = cv2.flip(frame, 1)
    
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.4*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    roi = frame[y1:y2, x1:x2]
    
    test_image=processing(roi)
    print(test_image)
    cv2.imwrite(path,test_image)
    #result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
    #print(result)
    cv2.imshow("frame",frame)
    key=cv2.waitKey(20)
    if(key==ord('q')):
        break;
        
cap.release()
cv2.destroyAllWindows()        