#import cv2
import numpy as np
import os
import h5py

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils import data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.backends import cudnn

from PIL import Image
from tqdm import tqdm 
import random 

import torchvision
import torchvision.transforms as transforms
from torch import nn, optim

def sample_from_celebA(num_samples_desired):
    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)


    # Change this part to locate the correct directory where dataset resides
    use_colab = False

    if use_colab:
        from google.colab import drive
        drive.mount('/content/gdrive')
        #!ls "/content/gdrive/My Drive/Datasets/shared/HW4_shared_files"

        ## Update the experiments directory
        EXPERIMENTS_DIRECTORY = '/content/gdrive/My Drive/Datasets/shared/experiments/'
        DATA_DIRECTORY = '/content/gdrive/My Drive/Datasets/shared/HW4_shared_files/'
        CELEBA_GOOGLE_DRIVE_PATH = DATA_DIRECTORY + 'celeba_attributes_images.hdf5'
        IMDB_REVIEWS_FILE_PATH = DATA_DIRECTORY + 'data/'
    else:
        ## Update the experiments directory
        #EXPERIMENTS_DIRECTORY = '/projectnb/dl523/students/mqraitem/experiments/'
        DATA_DIRECTORY = '/projectnb/dl523/materials/datasets/'
        CELEBA_GOOGLE_DRIVE_PATH = DATA_DIRECTORY + 'celeba_attributes_images.hdf5'
        IMDB_REVIEWS_FILE_PATH = DATA_DIRECTORY + 'data/'



    cudnn.benchmark = True

    def get_experiment_configuration(repeat_num=1,num_iters=200000, 
                  batch_size=5000, mode='train', resume_iters=False,
                  selected_attributes = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']):
        config = {}

        # Model configurations.
        config['c_dim'] = len(selected_attributes)
        config['image_size'] = 64
        config['selected_attributes'] = selected_attributes 

        # Training configurations.
        config['batch_size'] = batch_size #16


        # Test configurations.
        config['test_iters'] = num_iters

        # Miscellaneous.
        config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config['num_workers'] = 1
        config['mode'] = mode

        # Step size.


        #EXPERIMENT_RESULTS_FOLDER = EXPERIMENTS_DIRECTORY + 'gan-experiments/'

        suffix = str(repeat_num) + '-cdim-' + str(len(selected_attributes))
        #config['log_dir'] = EXPERIMENT_RESULTS_FOLDER + 'logs-' + suffix
        #config['sample_dir'] = EXPERIMENT_RESULTS_FOLDER + 'sample_dir-' + suffix
        #config['model_save_dir'] = EXPERIMENT_RESULTS_FOLDER + 'model_save_dir-' + suffix
        #config['result_dir'] = EXPERIMENT_RESULTS_FOLDER + 'result_dir-' + suffix

        print('\n\nPlease ensure you are using a GPU for computation')
        print('Will be using the following device for computation : ', config['device'])

        # Create directories if not exist.
        #if not os.path.exists(config['log_dir']):
        #    os.makedirs(config['log_dir'])
        #if not os.path.exists(config['sample_dir']):
        #    os.makedirs(config['sample_dir'])
        #if not os.path.exists(config['model_save_dir']):
        #    os.makedirs(config['model_save_dir'])
        #if not os.path.exists(config['result_dir']):
        #    os.makedirs(config['result_dir'])

        return config


    class CelebA(torch.utils.data.Dataset):
        """Dataset class for the CelebA dataset."""

        def __init__(self, transform, mode, config):
            """Initialize and preprocess the CelebA dataset."""

            self.file = h5py.File(CELEBA_GOOGLE_DRIVE_PATH, 'r')
            self.total_num_imgs, self.H, self.W, self.C = self.file['images'].shape

            self.images = self.file['images']
            self.attributes = self.file['attributes']

            self.selected_attrs = config['selected_attributes'] 
            self.all_attr_names = ALL_ATTRIBUTES

            self.transform = transform
            self.mode = mode

            self.train_dataset = []
            self.test_dataset = []
            self.attr2idx = {}
            self.idx2attr = {}
            self.preprocess()

            if mode == 'train':
                self.num_images = len(self.train_dataset)
            else:
                self.num_images = len(self.test_dataset)

        def preprocess(self):
            """Preprocess the CelebA attribute file."""
            for i, attr_name in enumerate(self.all_attr_names):
                self.attr2idx[attr_name] = i
                self.idx2attr[i] = attr_name

            self.all_idxs = np.arange(self.total_num_imgs)
            N_test = 9
            self.train_dataset = self.all_idxs[:-N_test] 
            self.test_dataset = self.all_idxs[-N_test:]

            random.seed(1234)
            np.random.seed(1234)        
            np.random.shuffle(self.train_dataset)

            print('Finished preprocessing the CelebA dataset...')

        def __getitem__(self, index):
            """Return one image and its corresponding attribute label."""
            dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
            idx = dataset[index]

            image = self.file['images'][idx]
            attributes = self.file['attributes'][idx]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(attributes[idx])
            
            return self.transform(image), torch.FloatTensor(label)

        def __len__(self):
            """Return the number of images."""
            return self.num_images


    def get_loader(config, mode='train'):
        """Build and return a data loader."""
        
        batch_size = config['batch_size']
        num_workers = config['num_workers']
        
        transform = []
        transform.append(T.ToPILImage())
        #if mode == 'train':
        #    transform.append(T.RandomHorizontalFlip())
        transform.append(T.ToTensor())
        #transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        
        dataset = CelebA(transform, mode, config)

        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=(mode=='train'),
                                      num_workers=num_workers)
        return data_loader
      
    def denorm(x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)



    ALL_ATTRIBUTES = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',  
          'Bags_Under_Eyes',  'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 
          'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
          'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
          'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 
          'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 
          'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
          'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
          'Wearing_Necklace', 'Wearing_Necktie', 'Young' ]

    small_config = get_experiment_configuration(repeat_num=1, num_iters=20000,
                  batch_size=num_samples_desired, selected_attributes = ALL_ATTRIBUTES)



    data_loader = get_loader(small_config, 'train')
    data_iter = iter(data_loader)
    x_fixed, c_org = next(data_iter)


    for i in range (x_fixed.size()[0]):
        x=x_fixed[i,:,:,:]
        x_gray=x[0,:,:]* 0.2989 + x[1,:,:]* 0.5870 + x[2,:,:]* 0.1140
        if i==0:
            gray_celeba=np.expand_dims(x_gray.numpy(),0) 
        #grays.append(x_gray.numpy())
        else:
            gray_celeba=np.concatenate([gray_celeba,np.expand_dims(x_gray.numpy(),0)])
        

    #plt.imshow(gray_celeba[0,:,:],cmap='gray')
    return gray_celeba







