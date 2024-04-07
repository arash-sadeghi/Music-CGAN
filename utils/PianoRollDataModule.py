import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from CONST_VARS import CONST
from utils.Utility_functions import  resize_to_batch_compatible,get_pianoroll_id_list,pianoroll2numpy
import numpy as np
import os
import torch
class PianoRollDataModule(pl.LightningDataModule):
    DATA_SAVE_URL = os.path.join(CONST.dataset_root,"data_genred.npy")
    GENRE_SAVE_URL = os.path.join(CONST.dataset_root,"genre.npy")
    def __init__(
        self,
        data_dir: str = CONST.dataset_root,
        batch_size: int = CONST.BATCH_SIZE,
        # num_workers: int = CONST.NUM_WORKERS,
        ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # self.num_workers = num_workers

        self.transform = transforms.Compose( #? what is this?
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10

        self.prepare_data()

    def prepare_data(self): #! 1 automatically called upon calling trainer.fit(model , dm) in main. this function is for downloading dataset        # download
        
        id_list ,  genres = get_pianoroll_id_list() #TODO bring implementation
        
        if os.path.exists(PianoRollDataModule.DATA_SAVE_URL):
            print("[+] loading data from existing npy file")
            self.load_data_np()
        else:
            print("[+] creating numpy dataset")
            self.data_np , self.genre_per_sample = pianoroll2numpy(id_list,genres)
            self.save_data_np()

        # draw_example_pianoroll(data)

        # self.rock_dataloader = self.generate_rock_dataloader() #TODO has error
        
        if self.data_np.shape[0]%CONST.BATCH_SIZE != 0:
            self.data_np , self.genre_per_sample= resize_to_batch_compatible(self.data_np , self.genre_per_sample)

        train_dataset = CustomDataset(drum = self.data_np[:,0,:,:].astype(np.float32) , bass = self.data_np[:,3,:,:].astype(np.float32) , genre = self.genre_per_sample)
        self.train_data_loader = CONST.torch.utils.data.DataLoader(train_dataset, batch_size=CONST.BATCH_SIZE, shuffle=True)
        
        print("Number of Batches:", len(self.train_data_loader))
    
    def generate_rock_dataloader(self):
        rock_measures_filter = self.data_np[self.genre_per_sample == CONST.genre_code['Pop_Rock']]
        rock_measures =  CustomDataset(drum = self.data_np[rock_measures_filter,0,:,:].astype(np.float32) , bass = self.data_np[rock_measures_filter,3,:,:].astype(np.float32) , genre = self.genre_per_sample[rock_measures_filter])   
        return CONST.torch.utils.data.DataLoader(rock_measures, batch_size=1, shuffle=False)
    
    def setup(self, stage=None): #! 2 automatically called upon calling trainer.fit(model , dm) in main , after execution of prepare_data. stage automatiically passed
        # # Assign train/val datasets for use in dataloaders
        # if stage == "fit" or stage is None:
        #     mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        #     self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # # Assign test dataset for use in dataloader(s)
        # if stage == "test" or stage is None:
        #     self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        pass

    def get_train_dataloader(self):
        return self.train_data_loader

    def get_rock_dataloader(self):
        return self.rock_dataloader

    def save_data_np(self):
        np.save(PianoRollDataModule.DATA_SAVE_URL , self.data_np)
        np.save(PianoRollDataModule.GENRE_SAVE_URL , self.genre_per_sample)

    def load_data_np(self):
        self.data_np = np.load(PianoRollDataModule.DATA_SAVE_URL)
        self.genre_per_sample = np.load(PianoRollDataModule.GENRE_SAVE_URL)
    
    #TODO write validator data part
        

class CustomDataset(Dataset):
    def __init__(self, drum , bass , genre):
        self.bass = bass
        self.drum = drum
        self.genre = genre

    def __len__(self):
        return len(self.bass)  # Assuming all input_data have the same length

    def __getitem__(self, idx):
        bass = torch.from_numpy(self.bass[idx])  # Convert to PyTorch tensor
        drum = torch.from_numpy(self.drum[idx])  # Convert to PyTorch tensor
        genre = torch.from_numpy(np.array(self.genre[idx]))  # Convert to PyTorch tensor
        return drum , bass , genre