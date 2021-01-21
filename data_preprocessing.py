# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 22:12:48 2021

@author: Yash
"""
import os
import cv2
from image_preprocessing import processing 
path=os.getcwd() + "\\asl_dataset"
path1=os.getcwd()+"\\data_set"

if not os.path.exists("data_set"):
    os.makedirs("data_set")
if not os.path.exists("data_set/train"):
    os.makedirs("data_set/train")
if not os.path.exists("data_set/test"):
    os.makedirs("data_set/test")
    
for (dirpath,dirnames,filenames) in os.walk(path):
    for dirname in dirnames:
        print(dirname)
        for(subDirpath,subDirname,subDirfile) in os.walk(path+"\\"+dirname):
            if not os.path.exists(path1+"/train/"+dirname):
                os.makedirs(path1+"/train/"+dirname)
            if not os.path.exists(path1+"/test/"+dirname):
                os.makedirs(path1+"/test/"+dirname)
                
            n=0.80*(len(subDirfile))
            
            i=0;
            for file in subDirfile:
                actual_path=path+"/"+dirname+"/"+file
                actual_path1=path1+"/"+"train/"+dirname+"/"+file
                actual_path2=path1+"/"+"test/"+dirname+"/"+file
                final_img=processing(actual_path)
                if i<n:
                    cv2.imwrite(actual_path1,final_img)
                else:
                    cv2.imwrite(actual_path2,final_img)
                    
                i=i+1
                
                
                
             
             

                 
                 
              
             
            
                 
                
             
             
             
         
         
         




