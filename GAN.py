import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchvision

from Generator import Generator
from Discriminator import Discriminator

from CONST_VARS import CONST
import os
from tqdm import tqdm
from IPython.display import clear_output
from pypianoroll import Multitrack, Track
import pypianoroll
import numpy as np
from Utility_functions import compute_gradient_penalty, display_pianoRoll
class GAN:
    def __init__(self,data_loader) -> None:
        self.discriminator = Discriminator() 
        self.generator = Generator() 
        self.data_loader = data_loader
        print(f"[+] is gpu availble {CONST.torch.cuda.is_available()}")
        self.running_d_loss, self.running_g_loss= 0.0, 0.0


    def train_one_step(self , real_samples):
        """Train the networks for one step."""
        # Sample from the lantent distribution
        latent = CONST.torch.randn(CONST.BATCH_SIZE, CONST.latent_dim) #! latent vector is always a random vector

        # Transfer data to GPU
        if CONST.torch.cuda.is_available():
            real_samples = real_samples.cuda()
            latent = latent.cuda()

        # === Train the discriminator ===
        ## train for real images 
        self.d_optimizer.zero_grad()
        ### Get discriminator outputs for the real samples
        prediction_real = self.discriminator(real_samples[0])
        ### Compute the loss function
        d_loss_real = -CONST.torch.mean(prediction_real)
        ### Backpropagate the gradients
        d_loss_real.backward()
        
        ## train for fake images
        ### Generate fake samples with the generator
        fake_samples = self.generator(latent,real_samples[0][:,1,:,:]) 
        ### Get discriminator outputs for the fake samples
        fake_samples_conditioned = CONST.torch.cat((fake_samples,real_samples[0][:,1,:,:].unsqueeze(1)),axis=1)
        prediction_fake_d = self.discriminator(fake_samples_conditioned.detach())
        ### Compute the loss function
        d_loss_fake = CONST.torch.mean(prediction_fake_d)
        ### Backpropagate the gradients
        d_loss_fake.backward()

        # Compute gradient penalty
        #! I Don't Know what does this do
        gradient_penalty = 10.0 * compute_gradient_penalty(
            self.discriminator, real_samples[0].data, fake_samples[0].data)
        # Backpropagate the gradients
        gradient_penalty.backward()

        # Update the weights
        self.d_optimizer.step()

        # === Train the generator ===
        # Reset cached gradients to zero
        self.g_optimizer.zero_grad()
        # Get discriminator outputs for the fake samples
        prediction_fake_g = self.discriminator(fake_samples_conditioned)
        # Compute the loss function
        g_loss = -CONST.torch.mean(prediction_fake_g)
        # Backpropagate the gradients
        g_loss.backward()
        # Update the weights
        self.g_optimizer.step()

        return d_loss_real + d_loss_fake, g_loss

    def train_prep(self):
        print("Number of parameters in G: {}".format(
            sum(p.numel() for p in self.generator.parameters() if p.requires_grad)))
        print("Number of parameters in D: {}".format(
            sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)))

        self.d_optimizer = CONST.torch.optim.Adam(self.discriminator.parameters(), lr=0.001,  betas=(0.5, 0.9))
        self.g_optimizer = CONST.torch.optim.Adam(self.generator.parameters(), lr=0.001, betas=(0.5, 0.9))

        # Prepare the inputs for the sampler, which wil run during the training
        self.sample_latent_eval = CONST.torch.randn(CONST.n_samples, CONST.latent_dim)

        # Transfer the neural nets and samples to GPU
        if CONST.torch.cuda.is_available():
            self.discriminator = self.discriminator.cuda()
            self.generator = self.generator.cuda()
            self.sample_latent_eval = self.sample_latent_eval.cuda()

    def train_loop(self):
        self.train_prep()

        step = 0
        progress_bar = tqdm(total=CONST.n_steps, initial=step, ncols=80, mininterval=1)

        for step in range(CONST.n_steps):
            for real_samples in self.data_loader: #! [batch_size , instruments, time, pitch]

                #! test
                # from Conditioner import Conditioner
                # c = Conditioner()
                # res = c(real_samples[0][:,0,:,:].unsqueeze(1)) #! add one dimention as a chanel after batch dimention
                # self.generator(CONST.torch.randn(1, CONST.latent_dim*2))
                #! test

                # Train the neural networks
                self.generator.train() #! put generator in train mode. why dont we do this to discriminator?

                d_loss, g_loss = self.train_one_step(real_samples)

                # Record smoothened loss values for logger
                self.smooth_loss(d_loss , g_loss)

                # Update losses to progress bar
                progress_bar.set_description_str(
                    "(d_loss={: 8.6f}, g_loss={: 8.6f})".format(d_loss, g_loss))

            if step % CONST.sample_interval == 0:
                self.generator_generate_sample_output(real_samples,step)

            progress_bar.update(1)
  
    def generator_generate_sample_output(self,real_samples,step):
        # Create an empty dictionary to sotre history samples
        history_samples = {}

        # Get generated samples
        self.generator.eval()

        samples = self.generator(self.sample_latent_eval,real_samples[0][0,1,:,:].unsqueeze(0))
        history_samples[step] = CONST.torch.cat((samples[0].cpu().detach(),real_samples[0][0,1,:,:].unsqueeze(0))).numpy()


        # Display loss curves
        clear_output(True)

        CONST.writer.add_scalar("g_loss" , self.running_g_loss , step)
        CONST.writer.add_scalar("d_loss" , -self.running_d_loss , step)

        display_pianoRoll(history_samples[step],step)
        

    def smooth_loss(self, d_loss , g_loss):
        self.running_d_loss = 0.05 * d_loss + 0.95 * self.running_d_loss
        self.running_g_loss = 0.05 * g_loss + 0.95 * self.running_g_loss
