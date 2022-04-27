import os
import torch
import torch.nn as nn
import sys
sys.path.append("/Users/can/Documents/GitHub/dlsisr")
from data_loader import get_loader
from common.read_data import prepare_images_att
import numpy as np
from datasets import ATTImages as att
import matplotlib.pyplot as plt

config = {}
config['data_dir'] = 'data/ORL-DATABASE'
from common.disruptor import add_blur_decrease_size
#%%

script_location="/Users/can/Documents/GitHub/dlsisr"
os.chdir(script_location)

seen_people_tr, seen_people_te, unseen_people = prepare_images_att(config['data_dir'])
#train_loader, test_loader, validate_loader = get_loader(config, None)

mu = np.mean(seen_people_tr.flatten())/255.
sigma = np.std(seen_people_tr.flatten())/255.
#%%
train_data = att(seen_people_tr, mean=mu, std=sigma)


        

#%%


i=0
_,_,old_lr=add_blur_decrease_size(train_data.hr, (16, 16), add_blur=False)


plt.imshow(train_data.hr[i],cmap="gray")
#see that both train_data.lr and old_lr are similar to each other.
plt.imshow(train_data.lr[i],cmap="gray")
plt.imshow(old_lr[0],cmap="gray")

#%%



#%%


