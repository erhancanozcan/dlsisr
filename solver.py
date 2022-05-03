from model import Generator
from model import Discriminator
from model import VGG19Grayscale
from torchvision.utils import save_image
import torch
import os
import time
import datetime
import torch.nn as nn
import kornia
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from common.disruptor import downsampler

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, train_loader, test_loader, validate_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validate_loader = validate_loader

        # Model configurations.
        self.image_size = config['image_size']
        self.vgg_feature_layer = config['vgg_feature_layer']

        # Training configurations.
        self.batch_size = config['batch_size']
        self.num_iters = config['num_iters']
        self.num_iters_decay = config['num_iters_decay']
        self.g_lr = config['g_lr']
        self.d_lr = config['d_lr']
        self.n_critic = config['n_critic']
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        self.resume_iters = config['resume_iters']
        self.content_loss = config['content_loss']
        self.load_iters = config['load_iters']
        self.include_batch_norm = config['include_batch_norm']
        self.lambda_rec = config['lambda_rec']

        # Test configurations.
        self.mode = config['mode']

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config['log_dir']
        self.sample_dir = config['sample_dir']
        self.model_save_dir = config['model_save_dir']
        self.result_dir = config['result_dir']

        # Step size.
        self.log_step = config['log_step']
        self.sample_step = config['sample_step']
        self.model_save_step = config['model_save_step']
        self.lr_update_step = config['lr_update_step']
        
        # Downsampler to use in reconstruction loss
        self.ds=downsampler()

        # Build the model.
        self.build_model()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(in_channels=1, out_channels=1, include_batch_norm=self.include_batch_norm)
            
        if self.resume_iters:
            G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(self.load_iters))
            self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            
            
        self.D = Discriminator(in_channels=1)
        self.VGG19 = VGG19Grayscale(self.vgg_feature_layer)

        self.G, self.D, self.VGG19 = self.to_device(self.G, self.D, self.VGG19)


        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        
    def remove_batchnorm(self, x):
        if isinstance(x, nn.BatchNorm2d):
            x.reset_parameters()
            x.eval()
            with torch.no_grad():
                x.weight.fill_(1.0)
                x.bias.zero_()

    def to_device(self, *args):
        send = lambda x: x
        if self.device != 'cpu':
            dev = torch.device(self.device)
            send = lambda x: x.to(dev)
        return (send(x) for x in args)

    def plot_2_losses(self, loss1, loss2, name1, name2, title):
        # Losses
        plt.figure(figsize=(5,2.5))
        plt.title(title)
        plt.plot(loss1,label=name1)
        plt.plot(loss2,label=name2)
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, '{}.png'.format(title)))

    def plot_all_losses(self, train_loss, test_loss, valid_loss, name, title):
        # Losses
        plt.figure(figsize=(5,2.5))
        plt.title(title)
        plt.plot(train_loss,label='{}_train'.format(name))
        plt.plot(test_loss,label='{}_seen'.format(name))
        plt.plot(valid_loss,label='{}_unseen'.format(name))
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, '{}.png'.format(title)))

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def save_model(self, i):
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        print('Saved model checkpoints into {}...'.format(self.model_save_dir))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))


    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()


    def generator_loss(self, gen_hr, imgs_hr, valid, fake):
        # Losses
        mse = nn.MSELoss(reduction='sum').to(self.device)
        bce = nn.BCEWithLogitsLoss().to(self.device)
        l1 = nn.L1Loss()

        # Content loss
        if self.content_loss == 'mse':
            WH = imgs_hr.shape[2]*imgs_hr.shape[3]
            
            loss_content = mse(imgs_hr, gen_hr)/WH
        elif self.content_loss == 'vgg':
            gen_features = self.VGG19(gen_hr)
            real_features = self.VGG19(imgs_hr)

            WH = real_features.shape[2]*real_features.shape[3]
            loss_content = 0.006*mse(gen_features, real_features.detach())/WH

        # Adversarial loss
        N = imgs_hr.shape[0]
        loss_adversarial = bce(self.D(gen_hr), fake[:N])
        
        # Reconstruction loss
        loss_reconstruction = l1(self.ds(imgs_hr),self.ds(gen_hr))

        # Perceptual loss
        g_loss = loss_content + loss_adversarial + self.lambda_rec * loss_reconstruction

        return g_loss, loss_content, loss_adversarial, loss_reconstruction

    def discriminator_loss(self, imgs_lr, imgs_hr, valid, fake):
        bce = nn.BCEWithLogitsLoss(reduction='sum').to(self.device)

        # Loss of real and fake images
        dis_real = self.D(imgs_hr)
        dis_fake = self.D(self.G(imgs_lr).detach())

        N = imgs_hr.shape[0]
        is_valid = valid[:N]
        is_fake = fake[:N]

        d_loss = bce(torch.cat((dis_real, dis_fake)), torch.cat((is_valid, is_fake)))/N
        
        return d_loss

    def calculate_losses(self, gen_hr, imgs_hr):
        mse = nn.MSELoss(reduction='sum').to(self.device)
        l1 = nn.L1Loss().to(self.device)

        mse_value = mse(imgs_hr, gen_hr)
        l1_value = l1(imgs_hr, gen_hr)
        psnr_value = kornia.metrics.psnr(gen_hr, imgs_hr, max_val=1.0).to(self.device)
        ssim_value = kornia.metrics.ssim(imgs_hr, gen_hr, window_size=1).to(self.device)

        return mse_value.item(), l1_value.item(), psnr_value.item(), ssim_value.mean().item()

    def ave_losses(self, data_loader, gen_type = 'network'):

        MSE_losses = []
        L1_losses = []
        PSNR_losses = []
        SSIM_losses = []

        for i, imgs in enumerate(data_loader):
            # Prepare input images and target domain labels.
            imgs_lr = imgs["lr"].to(self.device)
            imgs_hr = imgs["hr"].to(self.device)

            if gen_type == 'bicubic':
                gen_hr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            else:
                gen_hr = self.G(imgs_lr).to(self.device)

            mse, l1, psnr, ssim = self.calculate_losses(gen_hr, imgs_hr)

            MSE_losses.append(mse)
            L1_losses.append(l1)
            PSNR_losses.append(psnr)
            SSIM_losses.append(ssim)

        mse_ave = np.mean(MSE_losses)
        l1_ave = np.mean(L1_losses)
        psnr_ave = np.mean(PSNR_losses)
        ssim_ave = np.mean(SSIM_losses)

        return mse_ave, l1_ave, psnr_ave, ssim_ave


    def train(self):
        """Train StarGAN within a single dataset."""

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Labels
        valid = 0.9*torch.ones(self.batch_size, 1, requires_grad=False)
        fake = torch.zeros(self.batch_size, 1, requires_grad=False)

        valid, fake = self.to_device(valid, fake)

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.load_iters

        # Start training.
        loss = {}
        G_losses = []
        D_losses = []

        content_losses = []
        advers_losses = []
        rec_losses = []

        MSE_losses = []
        L1_losses = []
        PSNR_losses = []
        SSIM_losses = []

        MSE_seen = []
        L1_seen = []
        PSNR_seen = []
        SSIM_seen = []

        MSE_unseen = []
        L1_unseen = []
        PSNR_unseen = []
        SSIM_unseen = []

        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            for j, imgs in enumerate(self.train_loader):
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                # Configure model input
                imgs_lr = imgs["lr"]
                imgs_hr = imgs["hr"]

                imgs_lr, imgs_hr = self.to_device(imgs_lr, imgs_hr)

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Backward and optimize.
                self.reset_grad()
                d_loss = self.discriminator_loss(imgs_lr, imgs_hr, valid, fake)
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss['D/loss'] = d_loss.item()

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #
                #if (i+1) % self.n_critic == 0:
                self.reset_grad()
                # Generate a high resolution image from low resolution input
                gen_hr = self.G(imgs_lr).to(self.device)

                # Backward and optimize.
                g_loss, loss_content, loss_adversarial, loss_reconstruction = self.generator_loss(gen_hr, imgs_hr, valid, fake)
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss'] = g_loss.item()
                loss['G/loss_content'] = loss_content.item()
                loss['G/loss_adversarial'] = loss_adversarial.item()
                loss['G/loss_reconstruction'] = loss_reconstruction.item()

                G_losses.append(g_loss)
                D_losses.append(d_loss)
                content_losses.append(loss_content)
                advers_losses.append(loss_adversarial)
                rec_losses.append(loss_reconstruction)


            # find losses for training data
            mse_train, l1_train, psnr_train, ssim_train = self.ave_losses(self.train_loader)
            MSE_losses.append(mse_train)
            L1_losses.append(l1_train)
            PSNR_losses.append(psnr_train)
            SSIM_losses.append(ssim_train)

            # find losses for seen data
            mse_seen, l1_seen, psnr_seen, ssim_seen = self.ave_losses(self.test_loader)
            MSE_seen.append(mse_seen)
            L1_seen.append(l1_seen)
            PSNR_seen.append(psnr_seen)
            SSIM_seen.append(ssim_seen)

            #find losses for unseen data
            mse_unseen, l1_unseen, psnr_unseen, ssim_unseen = self.ave_losses(self.validate_loader)
            MSE_unseen.append(mse_unseen)
            L1_unseen.append(l1_unseen)
            PSNR_unseen.append(psnr_unseen)
            SSIM_unseen.append(ssim_unseen)

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                # Save image grid with upsampled inputs and SRGAN outputs
                sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                imgs_bicub = nn.functional.interpolate(imgs_lr, scale_factor=4)
                # add padding to make image size from 16x16 to 64x64 for display
                tr = transforms.Pad(24)

                hr_range = (torch.min(imgs_hr), torch.max(imgs_hr))

                imgs_lr = make_grid(tr(imgs_lr), nrow=1, normalize=True, value_range=hr_range)
                imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True, value_range=hr_range)
                gen_hr = make_grid(gen_hr, nrow=1, normalize=True, value_range=hr_range)
                imgs_bicub = make_grid(imgs_bicub, nrow=1, normalize=True, value_range=hr_range)

                img_grid = torch.cat((imgs_lr, imgs_hr, imgs_bicub, gen_hr), -1)
                save_image(img_grid, sample_path, normalize=False)
                print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                self.save_model(i)

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

        # plot and save all produced losses
        self.plot_2_losses(G_losses, D_losses, "G", "D", "G_D loss")
        self.plot_2_losses(content_losses, advers_losses, "content", "adversarial", "con_adv_loss")
        self.plot_all_losses(MSE_losses, MSE_seen, MSE_unseen, "MSE", "MSE loss")
        self.plot_all_losses(L1_losses, L1_seen, L1_unseen, "L1", "L1 loss")
        self.plot_all_losses(PSNR_losses, PSNR_seen, PSNR_unseen, "PSNR", "PSNR loss")
        self.plot_all_losses(SSIM_losses, SSIM_seen, SSIM_unseen, "SSIM", "SSIM loss")

        array = np.transpose(np.array([G_losses, D_losses, content_losses, advers_losses, rec_losses]))
        np.savetxt(os.path.join(self.log_dir, 'losses_G_D.csv'), array, delimiter = ",", header="G, D, content, adver, recon")

        array = np.transpose(np.array([MSE_losses, MSE_seen, MSE_unseen]))
        np.savetxt(os.path.join(self.log_dir, 'losses_MSE.csv'), array, delimiter = ",", header="train, seen, unseen")
        
        array = np.transpose(np.array([L1_losses, L1_seen, L1_unseen]))
        np.savetxt(os.path.join(self.log_dir, 'losses_L1.csv'), array, delimiter = ",", header="train, seen, unseen")
        
        array = np.transpose(np.array([PSNR_losses, PSNR_seen, PSNR_unseen]))
        np.savetxt(os.path.join(self.log_dir, 'losses_PSNR.csv'), array, delimiter = ",", header="train, seen, unseen")
        
        array = np.transpose(np.array([SSIM_losses, SSIM_seen, SSIM_unseen]))
        np.savetxt(os.path.join(self.log_dir, 'losses_SSIM.csv'), array, delimiter = ",", header="train, seen, unseen")

        # find losses for bicubic
        mse_train, l1_train, psnr_train, ssim_train = self.ave_losses(self.train_loader, 'bicubic')
        mse_seen, l1_seen, psnr_seen, ssim_seen = self.ave_losses(self.test_loader, 'bicubic')
        mse_unseen, l1_unseen, psnr_unseen, ssim_unseen = self.ave_losses(self.validate_loader, 'bicubic')

        array = np.transpose(np.array([mse_train, mse_seen, mse_unseen]))
        np.savetxt(os.path.join(self.log_dir, 'losses_bicubic_MSE.csv'), array, delimiter = ",", header="train, seen, unseen")

        array = np.transpose(np.array([l1_train, l1_seen, l1_unseen]))
        np.savetxt(os.path.join(self.log_dir, 'losses_bicubic_L1.csv'), array, delimiter = ",", header="train, seen, unseen")

        array = np.transpose(np.array([psnr_train, psnr_seen, psnr_unseen]))
        np.savetxt(os.path.join(self.log_dir, 'losses_bicubic_PSNR.csv'), array, delimiter = ",", header="train, seen, unseen")

        array = np.transpose(np.array([ssim_train, ssim_seen, ssim_unseen]))
        np.savetxt(os.path.join(self.log_dir, 'losses_bicubic_SSIM.csv'), array, delimiter = ",", header="train, seen, unseen")



    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.load_iters)

        # Set data loader.

        if self.mode == 'test':
            data_loader = self.test_loader

        elif self.mode  == 'valid':
            data_loader = self.validate_loader

        with torch.no_grad():
            for i, imgs in enumerate(data_loader):

                # Prepare input images and target domain labels.
                imgs_lr = imgs["lr"].to(self.device)
                imgs_hr = imgs["hr"].to(self.device)
                gen_hr = self.G(imgs_lr).to(self.device)

                mse, l1, psnr, ssim = self.calculate_losses(gen_hr, imgs_hr)

                # Save the translated images.
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                imgs_bicub = nn.functional.interpolate(imgs_lr, scale_factor=4)
                # add padding to make image size from 16x16 to 64x64 for display
                tr = transforms.Pad(24)

                hr_range = (torch.min(imgs_hr), torch.max(imgs_hr))

                imgs_lr = make_grid(tr(imgs_lr), nrow=1, normalize=True, value_range=hr_range)
                imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True, value_range=hr_range)
                gen_hr = make_grid(gen_hr, nrow=1, normalize=True, value_range=hr_range)
                imgs_bicub = make_grid(imgs_bicub, nrow=1, normalize=True, value_range=hr_range)

                img_grid = torch.cat((imgs_lr, imgs_hr, imgs_bicub, gen_hr), -1)
                save_image(img_grid, result_path, normalize=False)
                print('Saved real and fake images into {}...'.format(result_path))



