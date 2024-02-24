from utils.PianoRollDataModule import PianoRollDataModule
from CONST_VARS import *
from GAN import GAN
if __name__ == '__main__': #! without this if statement you will see a  wierd error
    dm = PianoRollDataModule()
    model = GAN(dm.train_dataloader())

    model.train_loop()