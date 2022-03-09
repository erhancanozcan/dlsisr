import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import torch   


def prepare_images_att(path,p_seen_people=0.8,num_pic_tr=8,num_pic_te=2):
    #path = '/content/gdrive/Shareddrives/EC503 Project Team/Code/ORL-DATABASE'
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

    
    
    
    
    num_people=int(40*p_seen_people)
    
    
    ind=np.arange(num_people)
    train_ind=ind*10
    tmp=ind*10
    
    for i in range(num_pic_tr-1):
        train_ind=np.concatenate([train_ind,tmp+1+i])
        
    train_ind=np.sort(train_ind)
    
    seen_people_tr=X[train_ind,]    
    #np_training_class=y[train_ind,]
    
    
    ind=np.arange(num_people)
    test_ind=ind*10+num_pic_tr
    tmp=ind*10+num_pic_tr
    for i in range(num_pic_te-1):
        test_ind=np.concatenate([test_ind,tmp+1+i])
        
    test_ind=np.sort(test_ind)
    
    seen_people_te=X[test_ind,]    
    #np_test_class=y[test_ind,]
    
    ind=np.arange(num_people,40,1)
    unseen_ind=ind*10
    tmp=ind*10
    
    for i in range(9):
        unseen_ind=np.concatenate([unseen_ind,tmp+1+i])
        
    unseen_ind=np.sort(unseen_ind)
    
    unseen_people=X[unseen_ind,]    
    



    return seen_people_tr,seen_people_te,unseen_people