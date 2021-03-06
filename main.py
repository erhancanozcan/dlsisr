import os
import torch
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
os.system('pip3 install -r requirements.txt')

EXPERIMENTS_DIRECTORY = '/projectnb/dl523/projects/SRGAN/dlsisr-main/experiments/'

def str2bool(v):
    return v.lower() in ('true')

def main(config):

    # Data loader.
    print(config['mode'])
    train_loader, test_loader, validate_loader = get_loader(config, config['mode'])

    # Solver for training and testing StarGAN.
    solver = Solver(train_loader, test_loader, validate_loader, config)

    if config['mode'] == 'train':
        solver.train()

    elif config['mode'] == 'test':
        solver.test()

    elif config['mode'] == 'valid':
        solver.test()


def get_experiment_configuration(num_iters=200000,
          log_step=100, sample_step=100, model_save_step=10000,
          lr_update_step=100, batch_size=16, mode='train', content_loss='mse',
          resume_iters=False, load_iters = 300, vgg_layer = 28, include_batch_norm=True,
          lambda_rec=10):
    config = {}

    config['celebA'] = True # This is a flag controlling the which datasets to consider during training.
                             # If it is True, then we will consider celebA,too.

    config['num_samples_from_celebA'] = 2000  #how many images we want to sample from celebA

    config['data_dir'] = 'ORL-DATABASE'

    # Model configurations.
    config['image_size'] = 64


    # Training configurations.
    config['batch_size'] = batch_size #16
    config['num_iters'] = num_iters
    config['num_iters_decay'] = num_iters//2
    config['g_lr'] = 0.0001
    config['d_lr'] = 0.0001
    if content_loss == 'vgg':
        config['g_lr'] = 0.0001
        config['d_lr'] = 0.0001
    config['n_critic'] = 5
    config['beta1'] = 0.5 #v
    config['beta2'] = 0.999 #v
    config['resume_iters'] = resume_iters
    config['content_loss'] = content_loss
    config['vgg_feature_layer'] = vgg_layer
    config['load_iters'] = load_iters
    config['lambda_rec'] = lambda_rec

    # Miscellaneous.
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['num_workers'] = 1
    config['mode'] = mode
    config['include_batch_norm'] = include_batch_norm

    # Step size.
    config['log_step'] = log_step #10
    config['sample_step'] = sample_step
    config['model_save_step'] =  model_save_step #10000
    config['lr_update_step'] = lr_update_step # 1000

    EXPERIMENT_RESULTS_FOLDER = EXPERIMENTS_DIRECTORY + 'srgan-experiments/'

    config['log_dir'] = EXPERIMENT_RESULTS_FOLDER + 'logs'
    config['sample_dir'] = EXPERIMENT_RESULTS_FOLDER + 'sample_dir'
    config['model_save_dir'] = EXPERIMENT_RESULTS_FOLDER + 'model_save_dir'
    config['result_dir'] = EXPERIMENT_RESULTS_FOLDER + 'result_dir'

    if config['mode'] == 'test':
        config['result_dir'] = EXPERIMENT_RESULTS_FOLDER + 'test_result_dir'

    elif config['mode'] == 'valid':
        config['result_dir'] = EXPERIMENT_RESULTS_FOLDER + 'valid_result_dir'

    print('\n\nPlease ensure you are using a GPU for computation')
    print('Will be using the following device for computation : ', config['device'])

    # Create directories if not exist.
    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])
    if not os.path.exists(config['sample_dir']):
        os.makedirs(config['sample_dir'])
    if not os.path.exists(config['model_save_dir']):
        os.makedirs(config['model_save_dir'])
    if not os.path.exists(config['result_dir']):
        os.makedirs(config['result_dir'])

    return config


if __name__ == '__main__':
    # For fast training.
    cudnn.benchmark = True
    #run with mse for first 1000 iter
    config = get_experiment_configuration(num_iters=1000,
          log_step=100, sample_step=100, model_save_step=100,
          batch_size=32, mode='train', content_loss='mse',
          resume_iters=False, load_iters = 0, include_batch_norm=True, 
          lambda_rec=0)
    main(config)

    config = get_experiment_configuration(num_iters=1000,
          log_step=100, sample_step=100, model_save_step=100,
          batch_size=32, mode='test', content_loss='mse',
          resume_iters=True, load_iters = 1000, include_batch_norm=True,
          lambda_rec=0)
    main(config)

    config = get_experiment_configuration(num_iters=1000,
          log_step=100, sample_step=100, model_save_step=100,
          batch_size=32, mode='valid', content_loss='mse',
          resume_iters=True, load_iters = 1000, include_batch_norm=True,
          lambda_rec=0)
    main(config)

    #run with vgg
    config = get_experiment_configuration(num_iters=6000,
          log_step=100, sample_step=100, model_save_step=100,
          batch_size=32, mode='train', content_loss='vgg',
          resume_iters=True, load_iters = 1000, vgg_layer = 28,
          lambda_rec=0, include_batch_norm=True)
    main(config)

    config = get_experiment_configuration(num_iters=6000,
          log_step=100, sample_step=100, model_save_step=100,
          batch_size=32, mode='test', content_loss='vgg',
          resume_iters=False, load_iters=6000,
          lambda_rec=0, include_batch_norm=True)
    main(config)

    config = get_experiment_configuration(num_iters=6000,
          log_step=100, sample_step=100, model_save_step=100,
          batch_size=32, mode='valid', content_loss='vgg',
          resume_iters=False, load_iters = 6000,
          lambda_rec=0, include_batch_norm=True)
    main(config)

