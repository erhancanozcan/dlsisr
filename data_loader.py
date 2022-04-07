import sys
from torch.utils.data import DataLoader
sys.path.append('./common')
from common.read_data import prepare_images_att
from datasets import ATTImages as att

def get_loader(config, mode='train'):
    """Build and return a data loader."""
    
        #%% Create datasets and data loaders
    seen_people_tr, seen_people_te, unseen_people = prepare_images_att(config['data_dir'])
    
    train_data = att(seen_people_tr)
    test_data = att(seen_people_te)
    validate_data = att(unseen_people)
    
    train_loader = DataLoader(train_data, batch_size = config['batch_size'],
                              shuffle = True, num_workers = 1)
    test_loader = DataLoader(test_data, batch_size = config['batch_size'],
                              shuffle = False, num_workers = 1)
    validate_loader = DataLoader(validate_data, batch_size = config['batch_size'],
                              shuffle = False, num_workers = 1)

    return train_loader, test_loader, validate_loader