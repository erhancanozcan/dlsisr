import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import torch   


def prepare_images_att(num_people,num_pic_tr,num_pic_te):
    path = '/content/gdrive/Shareddrives/EC503 Project Team/Code/ORL-DATABASE'
    X = []
    y = []
    for i in range(40):
        for j in range(10):
            img = Image.open(os.path.join(path +'/s'+str(i+1), str(j+1)+'.pgm'))
            X.append(np.asarray(img, dtype=np.uint8).flatten())
            y.append(i)
    X = np.asarray(X) #All images
    y = np.asarray(y) #numbered people (40 in total)



    #Create the training and testing datasets
    width = 92
    height = 112
    
    
    X=X.reshape(400,height,width)

    
    
    
    
    
    ind=np.arange(num_people)
    train_ind=ind*10
    tmp=ind*10
    
    for i in range(num_pic_tr-1):
        train_ind=np.concatenate([train_ind,tmp+1+i])
        
    train_ind=np.sort(train_ind)
    
    np_training_input=X[train_ind,]    
    np_training_class=y[train_ind,]
    
    
    ind=np.arange(num_people)
    test_ind=ind*10+num_pic_tr
    tmp=ind*10+num_pic_tr
    for i in range(num_pic_te-1):
        test_ind=np.concatenate([test_ind,tmp+1+i])
        
    test_ind=np.sort(test_ind)
    
    np_test_input=X[test_ind,]    
    np_test_class=y[test_ind,]
    



    return width,height,np_training_input,np_test_input,np_training_class,np_test_class