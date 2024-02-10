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
from Utility_functions import compute_gradient_penalty
class GAN:
    def __init__(self,data_loader) -> None:
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.data_loader = data_loader
        print(f"[+] is gpu availble {CONST.torch.cuda.is_available()}")

    def train_one_step(self , d_optimizer, g_optimizer, real_samples):
        """Train the networks for one step."""
        # Sample from the lantent distribution
        latent = CONST.torch.randn(CONST.BATCH_SIZE, CONST.latent_dim) #! latent vector is always a random vector

        # Transfer data to GPU
        if CONST.torch.cuda.is_available():
            real_samples = real_samples.cuda()
            latent = latent.cuda()

        # === Train the discriminator ===
        ## train for real images 
        ### Reset cached gradients to zero
        d_optimizer.zero_grad()
        ### Get discriminator outputs for the real samples
        prediction_real = self.discriminator(real_samples)
        ### Compute the loss function
        ### d_loss_real = torch.mean(torch.nn.functional.relu(1. - prediction_real))
        d_loss_real = -CONST.torch.mean(prediction_real)
        ### Backpropagate the gradients
        d_loss_real.backward()
        
        ## train for fake images
        ### Generate fake samples with the generator
        fake_samples = self.generator(latent) #! torch.Size([1, 5, 64, 72])
        ### Get discriminator outputs for the fake samples
        prediction_fake_d = self.discriminator(fake_samples.detach())
        ### Compute the loss function
        ### d_loss_fake = torch.mean(torch.nn.functional.relu(1. + prediction_fake_d))
        d_loss_fake = CONST.torch.mean(prediction_fake_d)
        ### Backpropagate the gradients
        d_loss_fake.backward()

        # Compute gradient penalty
        gradient_penalty = 10.0 * compute_gradient_penalty(
            self.discriminator, real_samples.data, fake_samples.data)
        # Backpropagate the gradients
        gradient_penalty.backward()

        # Update the weights
        d_optimizer.step()

        # === Train the generator ===
        # Reset cached gradients to zero
        g_optimizer.zero_grad()
        # Get discriminator outputs for the fake samples
        prediction_fake_g = self.discriminator(fake_samples)
        # Compute the loss function
        g_loss = -CONST.torch.mean(prediction_fake_g)
        # Backpropagate the gradients
        g_loss.backward()
        # Update the weights
        g_optimizer.step()

        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer 

        return d_loss_real + d_loss_fake, g_loss

    def train_prep(self):
        discriminator = self.discriminator
        generator = self.generator
        print("Number of parameters in G: {}".format(
            sum(p.numel() for p in generator.parameters() if p.requires_grad)))
        print("Number of parameters in D: {}".format(
            sum(p.numel() for p in discriminator.parameters() if p.requires_grad)))

        # Create optimizers
        self.d_optimizer = CONST.torch.optim.Adam(
            discriminator.parameters(), lr=0.001,  betas=(0.5, 0.9))
        self.g_optimizer = CONST.torch.optim.Adam(
            generator.parameters(), lr=0.001, betas=(0.5, 0.9))

        # Prepare the inputs for the sampler, which wil run during the training
        self.sample_latent = CONST.torch.randn(CONST.n_samples, CONST.latent_dim)

        # Transfer the neural nets and samples to GPU
        if CONST.torch.cuda.is_available():
            discriminator = discriminator.cuda()
            generator = generator.cuda()
            self.sample_latent = self.sample_latent.cuda()

    def train_loop(self):
        print("\n[+] in training loop \n")
        self.train_prep()
        # Create an empty dictionary to sotre history samples
        history_samples = {}

        step = 0
        generator = self.generator
        train_one_step = self.train_one_step
        g_optimizer = self.g_optimizer
        d_optimizer = self.d_optimizer 
        # Create a progress bar instance for monitoring
        progress_bar = tqdm(total=CONST.n_steps, initial=step, ncols=80, mininterval=1)

        # Start iterations
        while step < CONST.n_steps + 1:
            # Iterate over the dataset
            for real_samples in self.data_loader:
                # Train the neural networks
                generator.train()
                d_loss, g_loss = train_one_step(d_optimizer, g_optimizer, real_samples[0])

                # Record smoothened loss values for logger
                if step > 0:
                    running_d_loss = 0.05 * d_loss + 0.95 * running_d_loss
                    running_g_loss = 0.05 * g_loss + 0.95 * running_g_loss
                else:
                    running_d_loss, running_g_loss = 0.0, 0.0

                # Update losses to progress bar
                progress_bar.set_description_str(
                    "(d_loss={: 8.6f}, g_loss={: 8.6f})".format(d_loss, g_loss))

                if step % CONST.sample_interval == 0:
                    # Get generated samples
                    generator.eval()
                    samples = generator(self.sample_latent).cpu().detach().numpy()
                    history_samples[step] = samples

                    # Display loss curves
                    clear_output(True)

                    CONST.writer.add_scalar("g_loss" , running_g_loss , step)
                    CONST.writer.add_scalar("d_loss" , -running_d_loss , step)

                    # Display generated samples
                    samples = samples.transpose(1, 0, 2, 3).reshape(CONST.n_tracks, -1, CONST.n_pitches)
                    tracks = []

                    for idx, (program, is_drum, track_name) in enumerate(zip(CONST.programs, CONST.is_drums, CONST.track_names)):
                        # pianoroll = np.pad(np.concatenate(data[:4], 1)[idx], ((0, 0), (lowest_pitch, 128 - lowest_pitch - n_pitches)))
                        pianoroll = np.pad(samples[idx] > 0.5,((0, 0), (CONST.lowest_pitch, 128 - CONST.lowest_pitch - CONST.n_pitches)))
                        tracks.append(Track(name=track_name,program=program,is_drum=is_drum,pianoroll=pianoroll))

                    m = Multitrack(tracks=tracks,tempo=CONST.tempo_array,resolution=CONST.beat_resolution)
                    #! save music to npz -> midi
                    m.save(os.path.join(CONST.training_output_path_root,str(step)+'.npz'))
                    tmp = pypianoroll.load(os.path.join(CONST.training_output_path_root,str(step)+'.npz'))
                    tmp.write(os.path.join(CONST.training_output_path_root,str(step)+'.midi'))

                    axs = m.plot()
                    plt.gcf().set_size_inches((16, 8))
                    for ax in axs:
                        for x in range(
                            CONST.measure_resolution,
                            CONST.n_samples * CONST.measure_resolution * CONST.n_measures,
                            CONST.measure_resolution
                        ):
                            if x % (CONST.measure_resolution * 4) == 0:
                                ax.axvline(x - 0.5, color='k')
                            else:
                                ax.axvline(x - 0.5, color='k', linestyle='-', linewidth=1)
                    plt.savefig(os.path.join(CONST.training_output_path_root,str(step)+'.png'))

                step += 1
                progress_bar.update(1)
                if step >= CONST.n_steps:
                    break

class GAN_MNIST(pl.LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim: int = 100,
        lr: float = 0.001,
        b1: float = 0.5,
        b2: float = 0.9,
        batch_size: int = CONST.BATCH_SIZE,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters() #! will make it accessable every where
        self.automatic_optimization = False

        self.generator = Generator(latent_dim=self.hparams.latent_dim) #! self.hparams.latent_dim gives access to parameters
        self.discriminator = Discriminator()

        self.validation_z = CONST.torch.randn(8, self.hparams.latent_dim) #! 8 images

        self.example_input_array = CONST.torch.zeros(2, self.hparams.latent_dim) #? why

    def forward(self, z): #! 4 automatically called after configure_optimizers
        return self.generator(z)

    def adverserial_loss(self, y_hat, y): #! yhat is predicted
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch): #! 5 we are implementing this function from pythorch lightining. everything will be taken care for us.
        #! DEBUG -----------------------------------------------------------------------
        # imgs, _ = batch # imgs: torch.Size([128 (batch size), 1, 28, 28])
        imgs= batch[0] # imgs: torch.Size([128 (batch size), 1, 28, 28])

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = CONST.torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g) #! inherited
        self.generated_imgs = self(z)

        # log sampled images
        sample_imgs = self.generated_imgs[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, 0)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = CONST.torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        #! how well we can judge real
        # adversarial loss is binary cross-entropy
        g_loss = self.adverserial_loss(self.discriminator(self(z)), valid)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g) #! thats wierd. maybe using tensorflow is better idea

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = CONST.torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        real_loss = self.adverserial_loss(self.discriminator(imgs), valid)

        # how well can it label as fake?
        fake = CONST.torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.adverserial_loss(self.discriminator(self(z).detach()), fake) #! the reason behind detach is that self(z) was calculated in previous if. We dont want to do that again so we call detach function. it detaches from computation graph

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def configure_optimizers(self): #! 3 automatically called after setup(self, stage=None): in MNISTDataModule
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = CONST.torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = CONST.torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], [] #! goes into self.optimizers()

    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images_at_the_end_of_validation", grid, self.current_epoch)
        print("---> on_validation_epoch_end")

    def plot_imgs(self, label = ""): # custom function
        z = self.validation_z.type_as(self.generator.lin1.weight) # move to gpu (or not)
        sample_imgs = self(z).cpu() # self(z) is forward pass thats why we need it in GPU

        if not self.logger is None:
          #! add images to tensorboard
          grid = torchvision.utils.make_grid(sample_imgs)
          self.logger.experiment.add_image("generated_images", grid, self.current_epoch)


        print('epoch' , self.current_epoch)
        fig = plt.figure()

        for _ in range(sample_imgs.size(0)):
            plt.subplot(2,4,_+1)
            plt.tight_layout()
            plt.imshow(sample_imgs.detach()[_,0,:,:] , cmap = 'gray_r' , interpolation='none')
            plt.title("generated data")
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')

        plt.savefig( os.path.join(CONST.outputs_url,"GAN_generated_output_sample_"+label+".png") )

    def on_epoch_end(self): #! automatically called
        print("---> on_epoch_end")
        self.plot_imgs()
