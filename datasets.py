import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torchvision.transforms as transforms

#%% Import common data package
import sys
sys.path.append('./common')

from common.disruptor import add_blur_decrease_size
from common.disruptor import downsampler

#%% Custom ATT dataset loader

class ATTImages(Dataset):
    def __init__(self, people, hr_resize_dim=(64, 64), lr_desired_dim=(16, 16), mean=0, std=1,celebA_data=None):
        self.people = people
        
        # Normalize grayscale images
        self.normalize = transforms.Normalize(mean, std)
        self.tt = transforms.ToTensor()

        self.resize_dim = hr_resize_dim
        self.desired_dim = lr_desired_dim

        self.ds = downsampler()

        _, _, self.hr = add_blur_decrease_size(self.people, self.resize_dim, add_blur=False)
        if celebA_data is None:
            pass
        else:
            #convert celebA data from float32 to uint8
            celebA_uint8 = (celebA_data*256).astype(np.uint8)
            #plt.imshow(celebA_uint8[1,:,:],cmap='gray')
            # merge 2 datasets together
            self.hr=np.concatenate([self.hr,celebA_uint8])
        #_, _, self.lr = add_blur_decrease_size(self.hr, self.desired_dim, add_blur=False)
        self.lr = self.ds(self.hr)

    def __getitem__(self, index):
        lr = self.normalize(self.lr[index]/255.) # downsampler produces pytorch tensors
        hr = self.normalize(self.tt(self.hr[index])) # hr images are PIL images
        return {"lr": lr, "hr": hr}

    def __len__(self):
        return len(self.hr)
