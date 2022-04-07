#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 04:50:29 2021

@author: can
"""
import cv2
import numpy as np

def add_blur_decrease_size_mod(np_training_input,np_test_input,desired_dim):
    
    
    no_of_training=len(np_training_input)
    no_of_test=len(np_test_input)
    
    training_input=[None]*no_of_training
    test_input=[None]*no_of_test
    
    for i in range(no_of_training):
        tr=np_training_input[i,]
        gausBlur = cv2.GaussianBlur(tr, (5,5),0)
        #add blur by averaging
        #gausBlur=cv2.blur(tr,(3,3))
        resized = cv2.resize(gausBlur, desired_dim)
        training_input[i]=resized
    
    for i in range(no_of_test):
        tr=np_test_input[i,]
        gausBlur = cv2.GaussianBlur(tr, (5,5),0)
        #add blur by averaging
        #gausBlur=cv2.blur(tr,(3,3))
        resized = cv2.resize(gausBlur, desired_dim)
        test_input[i]=resized
        
    
    width=desired_dim[0]
    height=desired_dim[1]
    training_input=np.array(training_input)
    test_input=np.array(test_input)
    
    return width,height,training_input,test_input



def add_blur_decrease_size(np_training_input,desired_dim,add_blur=False):
    
    
    no_of_training=len(np_training_input)
    
    
    training_input=[None]*no_of_training
    
    
    for i in range(no_of_training):
        tr=np_training_input[i,]
        if add_blur==True:
            tr = cv2.GaussianBlur(tr, (3,3),0)
        #add blur by averaging
        #gausBlur=cv2.blur(tr,(3,3))
        resized = cv2.resize(tr, desired_dim)
        training_input[i]=resized
        
    
    width=desired_dim[0]
    height=desired_dim[1]
    training_input=np.array(training_input)
    
    
    return width,height,training_input