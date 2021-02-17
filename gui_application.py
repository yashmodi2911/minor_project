import numpy as np
import cv2
from UI_processiong import processing 
cap=cv2.VideoCapture(0)
hand_cascade = cv2.CascadeClassifier('hand.xml')
while(cap.isOpened()):
    ret,frame=cap.read()
   
    cv2.imshow('output',frame)
    
    if(cv2.waitKey(1)& 0xFF ==ord('q')):
        break;
        
cap.release();
cv2.destroyAllWindows()