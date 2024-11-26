from models.Generator import Generator
from models.Discriminator import Discriminator
from CONST_VARS import CONST
from tqdm import tqdm
from IPython.display import clear_output
from utils.Utility_functions import compute_gradient_penalty, display_pianoRoll, print_ram_usage, get_time_name,create_path_if_not_exists
from utils.Draw_hist import draw_hist
import wandb
import os

class GAN:
    REAL_LABEL = 1
    FAKE_LABEL = 0
    COMMENT = 'high_res_measure_sigmoid'
    training_output_path_root = os.path.join('data','PianoRoll','results','genre','training_output_path_root',COMMENT+get_time_name())

    def __init__(self,train_dataloader) -> None:
        self.discriminator = Discriminator() 
        self.generator = Generator() 
        self.train_dataloader = train_dataloader
        print(f"[+] is gpu availble {CONST.torch.cuda.is_available()}")
        self.running_d_loss, self.running_g_loss= 0.0, 0.0
        wandb.init(project="Music-CGAN")
        wandb.run.name += GAN.COMMENT+get_time_name()
        wandb.watch(self.generator)
        wandb.watch(self.discriminator)
        create_path_if_not_exists(GAN.training_output_path_root)

        self.criterion = CONST.torch.nn.BCELoss()
        ngpu = 1
        self.device = CONST.torch.device("cuda:0" if (CONST.torch.cuda.is_available() and ngpu > 0) else "cpu")
        # self.device = CONST.torch.device("cpu")
        print('[GAN] device',self.device)

    def train_one_step(self , real_samples):
        latent = CONST.torch.randn(CONST.BATCH_SIZE, CONST.latent_dim) 

        # Transfer data to GPU
        if self.device.type == 'cuda':
            real_samples = [_.cuda() for _ in real_samples]
            latent = latent.cuda()

        # === Train the discriminator ===
        ## train for real images 
        self.d_optimizer.zero_grad()
        ### Get discriminator outputs for the real samples

        #! normalization for non-binary
        real_samples_normalized = [[],[]]
        if CONST.torch.max(real_samples[0])> 1 or CONST.torch.max(real_samples[1])> 1:
            real_samples_normalized[0] = real_samples[0]/127
            real_samples_normalized[1] = real_samples[1]/127
            assert CONST.torch.all(real_samples_normalized[0]>=0) and CONST.torch.all(real_samples_normalized[0]<=1) 
            assert CONST.torch.all(real_samples_normalized[1]>=0) and CONST.torch.all(real_samples_normalized[1]<=1) 

        drum_and_bass = CONST.torch.cat((real_samples_normalized[0].unsqueeze(1) , real_samples_normalized[1].unsqueeze(1)),axis=1)
        genre = real_samples[2]
        prediction_real = self.discriminator(drum_and_bass, genre)
        ### Compute the loss function
        d_loss_real = -CONST.torch.mean(prediction_real)

        # label = CONST.torch.full((CONST.BATCH_SIZE,), GAN.REAL_LABEL, dtype=CONST.torch.float, device=self.device)
        # d_loss_real = self.criterion(prediction_real.squeeze(), label)
        # d_loss_real_mean = d_loss_real.mean().item()
        ### Backpropagate the gradients
        d_loss_real.backward()
        
        ## train for fake images
        ### Generate fake samples with the generator
        fake_samples = self.generator(latent,drum_and_bass[:,1,:,:] , genre) 
        ### Get discriminator outputs for the fake samples
        fake_samples_conditioned = CONST.torch.cat((fake_samples, drum_and_bass[:,1,:,:].unsqueeze(1)),axis=1)
        prediction_fake_d = self.discriminator(fake_samples_conditioned.detach() , genre)
        ### Compute the loss function
        d_loss_fake = CONST.torch.mean(prediction_fake_d) #* discriminator will try to make this zero, thus learning to give zero label to fake data
        
        # label.fill_(GAN.FAKE_LABEL)
        # d_loss_fake = self.criterion(prediction_fake_d.squeeze(), label)
        # d_loss_fake_mean = d_loss_fake.mean().item()
        ### Backpropagate the gradients
        d_loss_fake.backward()

        # Compute gradient penalty
        #! I Don't Know what does this do
        gradient_penalty = 10.0 * compute_gradient_penalty(
            self.discriminator, drum_and_bass.data, fake_samples_conditioned.data, genre, self.device)
        # Backpropagate the gradients
        gradient_penalty.backward()

        # Update the weights
        self.d_optimizer.step()

        # === Train the generator ===
        # Reset cached gradients to zero
        self.g_optimizer.zero_grad()
        # Get discriminator outputs for the fake samples
        prediction_fake_g = self.discriminator(fake_samples_conditioned,genre)
        # Compute the loss function
        g_loss_fake = -CONST.torch.mean(prediction_fake_g) #* will try to generate more negative values, thus bigger predcition_fake_g values, thus discriminator is fooled
        
        # label.fill_(GAN.REAL_LABEL)
        # g_loss_fake = self.criterion(prediction_fake_g.squeeze(), label)
        # g_loss_fake_mean = g_loss_fake.mean().item()

        # Backpropagate the gradients
        g_loss_fake.backward()
        # Update the weights
        self.g_optimizer.step()

        # return d_loss_real_mean +d_loss_fake_mean, g_loss_fake_mean
        return d_loss_real.detach() + d_loss_fake.detach(), g_loss_fake.detach()

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
        if self.device.type == 'cuda':
            self.discriminator = self.discriminator.cuda()
            self.generator = self.generator.cuda()
            self.sample_latent_eval = self.sample_latent_eval.cuda()

    def train_loop(self):
        self.train_prep()

        step = 0
        progress_bar = tqdm(total=CONST.n_steps, initial=step, ncols=100, mininterval=1)

        n_batches = 16448//CONST.BATCH_SIZE #! cheated numbers
        while step < CONST.n_steps:
            batch_count = 0
            for real_samples in self.train_dataloader:

                self.generator.train() #! put generator in train mode. why dont we do this to discriminator?

                d_loss, g_loss = self.train_one_step(real_samples)

                # Record smoothened loss values for logger
                self.smooth_loss(d_loss , g_loss)

                # Update losses to progress bar
                progress_bar.set_description_str(
                    "(d_loss={: 8.6f}, g_loss={: 8.6f}), BP={: 2.2f}, RAM={: 2.2f}".format(d_loss, g_loss, batch_count/n_batches*100,print_ram_usage()))
                
                wandb.log({"running_g_loss": self.running_g_loss,"running_d_loss": self.running_d_loss},step=step)
                batch_count +=1

            if step == 0:
                self.set_val_data(real_samples)

            if step % CONST.sample_interval == 0:
                self.generator_generate_sample_output(real_samples,step)
                self.save_weights(step)

                # wandb.log({"g_parameters": wandb.Histogram(self.generator.parameters())})
                # wandb.log({"d_parameters": wandb.Histogram(self.discriminator.parameters())})


            progress_bar.update(1)
            step +=1
  
    def set_val_data(self,real_samples):
        drum_and_bass = CONST.torch.cat((real_samples[0].unsqueeze(1) , real_samples[1].unsqueeze(1)),axis=1)
        self.genre_val = real_samples [2][:CONST.n_samples]
        self.bass_val = drum_and_bass[:CONST.n_samples,1,:,:].unsqueeze(1)
        self.drum_gt_val = drum_and_bass[:CONST.n_samples,0,:,:].unsqueeze(1)

        #! for normalization check
        assert CONST.torch.any(drum_and_bass>1) == True

        if self.device.type == 'cuda':
            self.genre_val= self.genre_val.cuda()
            self.bass_val = self.bass_val.cuda()
            self.drum_gt_val = self.drum_gt_val.cuda()

    def generator_generate_sample_output(self,real_samples,step):
        # Create an empty dictionary to sotre history samples
        # history_samples = {}

        # Get generated samples
        self.generator.eval()

        samples = self.generator(self.sample_latent_eval, self.bass_val , self.genre_val) 
        
        samples = (samples - samples.min()) / (samples.max() - samples.min()) #! re-normalization of G. some values of samples can be more than 1. 
        samples = samples*127 #! de-normalization
        draw_hist(samples , os.path.join(GAN.training_output_path_root,f'dist_{step}.png') )
        #* reshaping data inorder to be saved as image
        if step == 0: #* sample zero is ground truth
            temp = CONST.torch.cat((self.drum_gt_val.cpu().detach(),self.bass_val.cpu().detach()  ),axis = 1).numpy()
            draw_hist(self.drum_gt_val.cpu().detach() , os.path.join(GAN.training_output_path_root,f'dist__gt.png') )

        else: 
            temp = CONST.torch.cat((samples.cpu().detach(),self.bass_val.cpu().detach() ),axis = 1).numpy()

        temp = temp.transpose(1,0,2,3)
        temp = temp.reshape(temp.shape[0] , temp.shape[1] * temp.shape[2] , temp.shape[3])
        # history_samples[step] = temp


        # Display loss curves
        clear_output(True)

        # CONST.writer.add_scalar("g_loss" , self.running_g_loss , step)
        # CONST.writer.add_scalar("d_loss" , -self.running_d_loss , step)

        image_path = display_pianoRoll(temp,step,self.genre_val,GAN.training_output_path_root)
        wandb.log({f"sample_piano_roll": wandb.Image(image_path)},step=step)        

    def smooth_loss(self, d_loss , g_loss):
        self.running_d_loss = 0.05 * d_loss + 0.95 * self.running_d_loss
        self.running_g_loss = 0.05 * g_loss + 0.95 * self.running_g_loss

    def save_weights(self,step):
        CONST.torch.save(self.generator.state_dict(), os.path.join(GAN.training_output_path_root,f'generator_{step}.pth'))
        CONST.torch.save(self.discriminator.state_dict(), os.path.join(GAN.training_output_path_root,f'discriminator_{step}.pth'))

    def load_weights(self,G_path,D_path):
        self.generator.load_state_dict(CONST.torch.load(G_path))
        self.discriminator.load_state_dict(CONST.torch.load(D_path))

    
