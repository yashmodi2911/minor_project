# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 22:59:49 2021

@author: hp
"""

import tkinter as tk
import cv2
import os
from PIL import Image, ImageTk
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

path=os.getcwd()
class GUI:
    def __init__(self):
        self.cap=cv2.VideoCapture(0)
        self.video_panel_image= None
        self.filter_panel_image= None
        self.directory='model'
        
        self.json_file = open(self.directory+"/model-bw.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights(self.directory+"/model-bw.h5")
        
        
        
        
        self.root=tk.Tk()
        self.root.title("Sign to Text Converter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x850+0+0")
        
        #Heading
        self.heading = tk.Label(self.root,text = "ASL recognition using CNN",font=("Comic Sans MS",23,"bold"))
        self.heading.place(x=180,y = 5)
        
        #Video Panel For Camera Input
        self.video=tk.Label(self.root)
        self.video.place(x = 135, y = 60, width = 640, height = 640)
        
        #Filtered image Panel
        self.filter =tk.Label(self.root)
        self.filter.place(x = 460, y = 95, width = 310, height = 310)
        
        #Character Panel
        self.charpanel =tk.Label(self.root) # Current SYmbol
        self.charpanel.place(x = 500,y=640)
        #Character text
        self.char =tk.Label(self.root)
        self.char.place(x = 10,y = 640)
        self.char.config(text="Character :",font=("Courier",30,"bold"))
        
        #Word Panel
        self.wordpanel =tk.Label(self.root) 
        self.wordpanel.place(x = 220,y=700)
        #Word text
        self.word =tk.Label(self.root)
        self.word.place(x = 10,y = 700)
        self.word.config(text ="Word :",font=("Courier",30,"bold"))
        
        #Sentence Panel
        self.senpanel =tk.Label(self.root)
        self.senpanel.place(x = 350,y=760)
        self.sent =tk.Label(self.root)
        self.sent.place(x = 10,y = 760)
        self.sent.config(text ="Sentence :",font=("Courier",30,"bold"))
        self.video_loop()
        
        
    def destructor(self):
        print("Closing Application...")
        self.root.destroy() #Destroying Main Window
        self.cap.release() #Releasing Camera
        cv2.destroyAllWindows()
        
   
        
    def video_loop(self):
        ret, frame = self.cap.read()
        if ret:
             img = cv2.flip(frame, 1)
            #Flipping 2-D array
            #flipcode = 0, About x-axis
            #flipcode = 1, about y-axis
            #flipcode = -1, about both
            
             x1 = int(0.5*frame.shape[1])
             y1 = 10
             x2 = frame.shape[1]-10
             y2 = int(0.5*frame.shape[1])
             cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
             
             img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
             
             self.video_panel_image=Image.fromarray(img)
             imgtk = ImageTk.PhotoImage(image=self.video_panel_image)
            
            #Configuring Video Panel defined in __init__.
             self.video.imgtk = imgtk
             self.video.config(image=imgtk)
             
             img = img[y1:y2, x1:x2]
             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
             blur = cv2.GaussianBlur(gray,(5,5),2)
             # https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-2-adaptive-thresholding/
             th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            
            # https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-1-simple-thresholding/r
            #First argument is the source image, which should be a grayscale image.
             retval, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            # m,n=res.shape
            # img_new=np.zeros([m,n])
             #for i in range(1, m-1):
              #   for j in range(1, n-1):
               #      temp = [res[i-1, j-1], 
                #             res[i-1, j], 
                #              res[i-1, j + 1], 
                 #             res[i, j-1], 
                  #            res[i, j], 
                   #          res[i + 1, j-1], 
                     #         res[i + 1, j], 
                   #           res[i + 1, j + 1]]
                   #  temp = sorted(temp) 
                    # img_new[i, j]= temp[4] 
            
            
            
             #img_new=img_new.astype(np.uint8)
             #res=img
             self.predict(res)
            #retval is used in Otsu thresholding  image binarization -- https://medium.com/@hbyacademic/otsu-thresholding-4337710dc519
             '''For this, our cv2.threshold() function is used, but pass an extra flag, cv2.THRESH_OTSU. For threshold value, simply
                pass zero. Then the algorithm finds the optimal threshold value and returns you as the second output, retVal.
                If Otsu thresholding is not used, retVal is same as the threshold value you used.'''
            #res is our thresholded image
             self.filter_panel_image=Image.fromarray(res)
             imgtk=ImageTk.PhotoImage(image=self.filter_panel_image)
             
             #configuring filter panel defined in __init__
             self.filter.imgtk=imgtk
             self.filter.config(image=imgtk)
            
            
            
             
        self.root.after(20, self.video_loop)
          
            
    def predict(self,test_image):
        test_image = cv2.resize(test_image, (128,128))
        result = self.loaded_model.predict_classes(test_image.reshape(1, 128, 128, 1))
        
        print(result)
       
        
        
            
        
        
        
        
pba = GUI()
pba.root.mainloop()
        