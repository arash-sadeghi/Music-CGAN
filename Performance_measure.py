import numpy as np
import torch
from utils.PianoRollDataModule import PianoRollDataModule
from CONST_VARS import CONST
import os
from models.Generator import Generator
import tqdm
import matplotlib.pyplot as plt

WEIGTH_PATH = 'data/PianoRoll/results/weights/generator'

def calculate_DP(data , batch_size):
    tolerance = 0.1
    drum_pattern_mask = np.tile([1., tolerance], 8)
    drum_pattern_mask = np.tile(drum_pattern_mask, (batch_size, CONST.n_measures, 1))

    data = data.sum(axis=3) #* sum pitches
    masked_drum = data * drum_pattern_mask
    return masked_drum.mean().item()

def plot(DP_list , data_DP ,EB_list ,data_EB ):
    plt.plot(DP_list , label='generator DP', marker = 'o')
    plt.plot(data_DP , label='data DP', marker = 'o')
    plt.plot(EB_list , label='generator EB', marker = 'o')
    plt.plot(data_EB , label='data EB', marker = 'o')

    plt.legend()

    plt.show()


def calculate_EB(data):
    return 1 - torch.mean(torch.any(data.view(data.shape[0],data.shape[1],-1) > 0.5, dim=2).float()).item()

def evaluate_generator(dm):
    train_data = dm.get_train_dataloader()

    weight_list = os.listdir(WEIGTH_PATH)
    generator = Generator()

    DP_list = []
    EB_list = []
    for weight in tqdm.tqdm(weight_list):
        generator.load_state_dict(CONST.torch.load(os.path.join(WEIGTH_PATH,weight) , map_location=torch.device('cpu')))
        EB = 0
        DP = 0
        for count , data_point in enumerate(train_data): 
            latent = CONST.torch.randn(dm.batch_size, CONST.latent_dim) 
            drum_and_bass = CONST.torch.cat((data_point[0].unsqueeze(1) , data_point[1].unsqueeze(1)),axis=1)
            drum = generator(latent,drum_and_bass[:,1,:,:] , data_point[2]).detach().cpu() #TODO processing on CPU
            drum_measure_seperated  = drum.reshape(drum.shape[0], CONST.n_measures , CONST.measure_resolution   ,drum.shape[-1])
            EB += calculate_EB(drum_measure_seperated)
            DP += calculate_DP(drum_measure_seperated , dm.batch_size)

        count +=1 #* because count started from zero
        DP_list.append(round(DP/(count)*100,2))
        EB_list.append(round(EB/(count)*100,2))
        print(f"loop steps {count}")
        print(f"[+] generator data weight {weight}: train_data_DP {DP_list[-1]} , train_data_EB {EB_list[-1]}")

    return DP_list, EB_list

def evaluate_dataset(dm,data_length):
    train_data = dm.get_train_dataloader()
    EB = 0
    DP = 0

    for count , data_point in enumerate(train_data): 
        drum = data_point[0]
        drum_measure_seperated  = drum.reshape(drum.shape[0], CONST.n_measures , CONST.measure_resolution   ,drum.shape[2])
        EB += calculate_EB(drum_measure_seperated)
        DP += calculate_DP(drum_measure_seperated , dm.batch_size)

    count +=1 #* because count started from zero
    data_EB = [round(EB/(count)*100,2)]*data_length
    data_DP = [round(DP/(count)*100,2)]*data_length
    print(f"loop steps {count}")
    print(f"[+] original data: train_data_DP {data_DP[-1]} , train_data_EB {data_EB[-1]}")
    return data_DP , data_EB

def evaluate(dm):
    DP_list, EB_list = evaluate_generator(dm)
    data_DP , data_EB = evaluate_dataset(dm,len(EB_list))
    plot(DP_list , data_DP ,EB_list ,data_EB )
    

if __name__ == '__main__':
    dm = PianoRollDataModule(batch_size='HDD')
    evaluate(dm)
