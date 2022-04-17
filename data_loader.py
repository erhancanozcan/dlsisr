import sys, numpy as np
from torch.utils.data import DataLoader
sys.path.append('./common')
from common.read_data import prepare_images_att
from datasets import ATTImages as att

def get_loader(config, mode='train'):
    """Build and return a data loader."""

    #%% Create datasets and data loaders
    seen_people_tr, seen_people_te, unseen_people = prepare_images_att(config['data_dir'])

    # Normalize all data against the training dataset
    mu = np.mean(seen_people_tr.flatten())/255.
    sigma = np.std(seen_people_tr.flatten())/255.

    train_data = att(seen_people_tr, mean=mu, std=sigma)
    test_data = att(seen_people_te, mean=mu, std=sigma)
    validate_data = att(unseen_people, mean=mu, std=sigma)

    train_loader = DataLoader(train_data, batch_size = config['batch_size'],
                              shuffle = True, num_workers = 1)
    test_loader = DataLoader(test_data, batch_size = config['batch_size'],
                              shuffle = False, num_workers = 1)
    validate_loader = DataLoader(validate_data, batch_size = config['batch_size'],
                              shuffle = False, num_workers = 1)

    return train_loader, test_loader, validate_loader
