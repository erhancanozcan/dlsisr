import numpy as np


from torch.utils.data import Dataset
import torchvision.transforms as transforms

#%% Import common data package
import sys
sys.path.append('./common')

from common.disruptor import add_blur_decrease_size
from common.disruptor import downsampler


#%% Custom ATT dataset loader

class ATTImages(Dataset):
    def __init__(self, people, hr_resize_dim=(64, 64), lr_desired_dim=(16, 16), mean=0, std=1):
        self.people = people

        # Normalize grayscale images
        self.normalize = transforms.Compose([
                # transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        self.resize_dim = hr_resize_dim
        self.desired_dim = lr_desired_dim
        
        self.ds=downsampler()

        _, _, self.hr = add_blur_decrease_size(self.people, self.resize_dim, add_blur=False)
        #_, _, self.lr = add_blur_decrease_size(self.hr, self.desired_dim, add_blur=False)
        self.lr = self.ds(self.hr)

    def __getitem__(self, index):
        lr = self.normalize(self.lr[index])
        hr = self.normalize(self.hr[index])
        return {"lr": lr, "hr": hr}

    def __len__(self):
        return len(self.people)