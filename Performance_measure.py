import numpy as np
import torch
from utils.PianoRollDataModule import PianoRollDataModule
from CONST_VARS import CONST
def calculate_DP(data , batch_size):
    tolerance = 0.1
    drum_pattern_mask = np.tile([1., tolerance], 8)
    drum_pattern_mask = np.tile(drum_pattern_mask, (batch_size, CONST.n_measures, 1))

    DP = 0
    for count , data_point in enumerate(data): 
        drum = data_point[0]
        drum_measure_seperated  = drum.reshape(drum.shape[0], CONST.n_measures , CONST.measure_resolution   ,drum.shape[2])
        drum_measure_seperated = drum_measure_seperated.sum(axis=3) #* sum pitches
        masked_drum = drum_measure_seperated * drum_pattern_mask
        DP += masked_drum.mean().item()

    count +=1 #* because count started from zero
    print(f"loop steps {count}")
    return round(DP/(count)*100,2)

def calculate_EB(data):
    EB = 0
    for count , data_point in enumerate(data): 
        drum = data_point[0]
        drum_measure_seperated  = drum.reshape(drum.shape[0], CONST.n_measures , CONST.measure_resolution   ,drum.shape[2])
        EB += 1 - torch.mean(torch.any(drum_measure_seperated.view(drum_measure_seperated.shape[0],drum_measure_seperated.shape[1],-1) > 0.5, dim=2).float()).item()

    count +=1 #* because count started from zero
    print(f"loop steps {count}")
    return round(EB/(count)*100,2)

if __name__ == '__main__':
    dm = PianoRollDataModule(batch_size='HDD')
    train_data = dm.get_train_dataloader()

    train_data_DP = calculate_DP(train_data , batch_size = dm.batch_size)
    print(f" train_data_DP {train_data_DP}")

    train_data_EB = calculate_EB(train_data)
    print(f" train_data_EB {train_data_EB}")
    
