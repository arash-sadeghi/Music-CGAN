import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from CONST_VARS import CONST
from Utility_functions import  draw_example_pianoroll,get_pianoroll_id_list,pianoroll2numpy
import numpy as np

class PianoRollDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = CONST.PATH_DATASETS,
        batch_size: int = CONST.BATCH_SIZE,
        num_workers: int = CONST.NUM_WORKERS,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

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
        
        id_list = get_pianoroll_id_list() #TODO bring implementation
        data = pianoroll2numpy(id_list)
        
        # draw_example_pianoroll(data)

        drum_and_bass = data[:,[0,3],:,:]
        drum_and_bass = np.repeat(drum_and_bass,100,axis=0) #! DEBUG repeating samples to imitate batch

        drum_and_bass_tensor = CONST.torch.as_tensor(drum_and_bass, dtype=CONST.torch.float32)
        dataset = CONST.torch.utils.data.TensorDataset(drum_and_bass_tensor) #! torch.Size([8, 5, 64, 72])
        # self.data_loader = CONST.torch.utils.data.DataLoader(dataset, batch_size=CONST.BATCH_SIZE, drop_last=True, shuffle=True)
        #! DEBUG: no shuffle:
        self.data_loader = CONST.torch.utils.data.DataLoader(dataset, batch_size=CONST.BATCH_SIZE, shuffle=False)
        
        print("Number of Batches:", len(self.data_loader))
    
    def setup(self, stage=None): #! 2 automatically called upon calling trainer.fit(model , dm) in main , after execution of prepare_data. stage automatiically passed
        # # Assign train/val datasets for use in dataloaders
        # if stage == "fit" or stage is None:
        #     mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        #     self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # # Assign test dataset for use in dataloader(s)
        # if stage == "test" or stage is None:
        #     self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        pass

    def train_dataloader(self):
        #! DEBUG --------------------------------------------------------------------------------------------------
        print("[!!!!!] Debug overfitting ")
        return self.data_loader

    # def val_dataloader(self):
    #     return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)