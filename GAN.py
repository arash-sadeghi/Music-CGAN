import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchvision

from Generator import Generator
from Discriminator import Discriminator
from Conditioner import Conditioner

from CONST_VARS import CONST
import os
from tqdm import tqdm
from IPython.display import clear_output
from pypianoroll import Multitrack, Track
import pypianoroll
import numpy as np
from Utility_functions import compute_gradient_penalty
class GAN:
    def __init__(self,data_loader) -> None:
        self.discriminator = Discriminator() 
        self.generator = Generator() 
        self.conditioner = Conditioner()
        self.data_loader = data_loader
        print(f"[+] is gpu availble {CONST.torch.cuda.is_available()}")
        self.running_d_loss, self.running_g_loss , self.running_c_loss = 0.0, 0.0, 0.0


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
        condition = self.conditioner.forward(real_samples[0][:,1,:,:]) #! get bass as condition
        conditioned_input = CONST.torch.cat((latent, condition),axis=1)
        fake_samples = self.generator(conditioned_input) 
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

        # === Train the conditioner ===
        #! I am not sure if this will work this way
        self.c_optimizer.zero_grad()
        self.c_optimizer.step()
        c_loss = g_loss #! for now conditioner shares the same loss with generator

        return d_loss_real + d_loss_fake, g_loss , c_loss

    def train_prep(self):
        print("Number of parameters in G: {}".format(
            sum(p.numel() for p in self.generator.parameters() if p.requires_grad)))
        print("Number of parameters in D: {}".format(
            sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)))
        print("Number of parameters in C: {}".format(
            sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad)))

        self.d_optimizer = CONST.torch.optim.Adam(self.discriminator.parameters(), lr=0.001,  betas=(0.5, 0.9))
        self.g_optimizer = CONST.torch.optim.Adam(self.generator.parameters(), lr=0.001, betas=(0.5, 0.9))
        self.c_optimizer = CONST.torch.optim.Adam(self.conditioner.parameters(), lr=0.001, betas=(0.5, 0.9))

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
                self.conditioner.train()

                d_loss, g_loss, c_loss = self.train_one_step(real_samples)

                # Record smoothened loss values for logger
                self.smooth_loss(d_loss , g_loss, c_loss)

                # Update losses to progress bar
                progress_bar.set_description_str(
                    "(d_loss={: 8.6f}, g_loss={: 8.6f}, c_loss={: 8.6f})".format(d_loss, g_loss, c_loss))

                if step % CONST.sample_interval == 0:
                    self.generator_generate_sample_output(real_samples,step)

                progress_bar.update(1)
  
    def generator_generate_sample_output(self,real_samples,step):
        # Create an empty dictionary to sotre history samples
        history_samples = {}

        # Get generated samples
        self.generator.eval()
        self.conditioner.eval()

        condition = self.conditioner.forward(real_samples[0][0,1,:,:].unsqueeze(0)) #! get bass as condition
        conditioned_input = CONST.torch.cat((self.sample_latent_eval, condition),axis = 1)
        samples = self.generator(conditioned_input).cpu().detach().numpy()
        history_samples[step] = samples

        # Display loss curves
        clear_output(True)

        CONST.writer.add_scalar("g_loss" , self.running_g_loss , step)
        CONST.writer.add_scalar("d_loss" , -self.running_d_loss , step)
        CONST.writer.add_scalar("c_loss" , self.running_c_loss , step)

        # # Display generated samples
        # samples = samples.transpose(1, 0, 2, 3).reshape(CONST.n_tracks, -1, CONST.n_pitches)
        # tracks = []

        # for idx, (program, is_drum, track_name) in enumerate(zip(CONST.programs, CONST.is_drums, CONST.track_names)):
        #     # pianoroll = np.pad(np.concatenate(data[:4], 1)[idx], ((0, 0), (lowest_pitch, 128 - lowest_pitch - n_pitches)))
        #     pianoroll = np.pad(samples[idx] > 0.5,((0, 0), (CONST.lowest_pitch, 128 - CONST.lowest_pitch - CONST.n_pitches)))
        #     tracks.append(Track(name=track_name,program=program,is_drum=is_drum,pianoroll=pianoroll))

        # m = Multitrack(tracks=tracks,tempo=CONST.tempo_array,resolution=CONST.beat_resolution)
        # #! save music to npz -> midi
        # m.save(os.path.join(CONST.training_output_path_root,str(step)+'.npz'))
        # tmp = pypianoroll.load(os.path.join(CONST.training_output_path_root,str(step)+'.npz'))
        # tmp.write(os.path.join(CONST.training_output_path_root,str(step)+'.midi'))

        # axs = m.plot()
        # plt.gcf().set_size_inches((16, 8))
        # for ax in axs:
        #     for x in range(
        #         CONST.measure_resolution,
        #         CONST.n_samples * CONST.measure_resolution * CONST.n_measures,
        #         CONST.measure_resolution
        #     ):
        #         if x % (CONST.measure_resolution * 4) == 0:
        #             ax.axvline(x - 0.5, color='k')
        #         else:
        #             ax.axvline(x - 0.5, color='k', linestyle='-', linewidth=1)
        # plt.savefig(os.path.join(CONST.training_output_path_root,str(step)+'.png'))

    def smooth_loss(self, d_loss , g_loss, c_loss):
        self.running_d_loss = 0.05 * d_loss + 0.95 * self.running_d_loss
        self.running_g_loss = 0.05 * g_loss + 0.95 * self.running_g_loss
        self.running_c_loss = 0.05 * c_loss + 0.95 * self.running_c_loss
