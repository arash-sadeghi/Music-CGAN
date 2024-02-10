# from MNISTDataModule import MNISTDataModule 
from PianoRollDataModule import PianoRollDataModule
from CONST_VARS import *
from GAN import GAN
# import pytorch_lightning as pl

if __name__ == '__main__': #! without this if statement you will see a  wierd error
    # dm = MNISTDataModule()
    dm = PianoRollDataModule()

    # model = GAN(*dm.dims)
    model = GAN(dm.train_dataloader())

    # model.plot_imgs("initial")

    # trainer = pl.Trainer(
    #     accelerator="auto",
    #     devices=1,
    #     max_epochs=5,
    # )

    # trainer.fit(model , dm)


    # model.plot_imgs("final")

    model.train_loop()