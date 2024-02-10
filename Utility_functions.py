import os
from CONST_VARS import CONST
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pypianoroll
from pypianoroll import Multitrack, Track
from tqdm import tqdm
import random

def compute_gradient_penalty( discriminator, real_samples, fake_samples): #! static method
    """Compute the gradient penalty for regularization. Intuitively, the
    gradient penalty help stablize the magnitude of the gradients that the
    discriminator provides to the generator, and thus help stablize the training
    of the generator."""
    # Get random interpolations between real and fake samples
    alpha = CONST.torch.rand(real_samples.size(0), 1, 1, 1).cuda() if CONST.torch.cuda.is_available() else CONST.torch.rand(real_samples.size(0), 1, 1, 1)

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    interpolates = interpolates.requires_grad_(True)
    # Get the discriminator output for the interpolations
    d_interpolates = discriminator(interpolates)
    # Get gradients w.r.t. the interpolations
    fake = CONST.torch.ones(real_samples.size(0), 1).cuda() if CONST.torch.cuda.is_available() else CONST.torch.ones(real_samples.size(0), 1)
    gradients = CONST.torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # Compute gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def is_folder_empty(folder_path):
    folder_size = sum(os.path.getsize(os.path.join(folder_path, file)) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file)))
    return folder_size <= 0

def convert_torch_dataset(subset):
    data_list = []
    for idx in range(len(subset)):
        # item = subset[idx]

        #! DEBUG
        item = subset[0] #! for overfitting
        
        data_list.append(item[0])  # Assuming each item is a tuple, and the data is at index 0

    numpy_array = np.array(data_list)
    plt.imshow(numpy_array[0][0])
    plt.title("over fitting instance")
    plt.savefig("over fitting instance.png")

    dataset = TensorDataset(CONST.torch.from_numpy(numpy_array))
    data_loader = DataLoader(dataset, batch_size=CONST.BATCH_SIZE, shuffle=True, num_workers=CONST.NUM_WORKERS)
    return data_loader

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def draw_example_pianoroll(data):
    #! example show
    tracks = []
    #! this loop will iterate 5 time: number of tracks
    for idx, (program, is_drum, track_name) in enumerate(zip(CONST.programs, CONST.is_drums, CONST.track_names)):
        #! np.concatenate(data[:4], 1) : arrays will be joined horizontally, [idx] selects the instrument, beats will be merged with samples thus one dimention reduction
        #! 4 samples * 64 notes each : 5 tracks 256 timesteps 72  pitches.
        pianoroll = np.pad(
            np.concatenate(data, 1)[idx], 
            ((0, 0), (CONST.lowest_pitch, 128 - CONST.lowest_pitch - CONST.n_pitches))
            )
        tracks.append(Track(name=track_name, program=program, is_drum=is_drum, pianoroll=pianoroll)) #! Track is from Pypianoroll
    multitrack = Multitrack(tracks=tracks, tempo=CONST.tempo_array, resolution=CONST.beat_resolution)
    axs = multitrack.plot()
    plt.gcf().set_size_inches((16, 8))
    #! this for loop only puts horizontal line in plot
    for ax in axs:
        for x in range(CONST.measure_resolution, data.shape[0]*data.shape[2], CONST.measure_resolution): #! data.shape[0]*data.shape[2]: samples * notes in each samples: whole notes
            if x % (CONST.measure_resolution * 4) == 0: #! marks 1 measure
                ax.axvline(x - 0.5, color='k')
            else:
                ax.axvline(x - 0.5, color='k', linestyle='-', linewidth=1) #! marks one beat of 4/4 (might include few notes)
    # plt.tight_layout()
    plt.savefig(CONST.example_dataset_path+".png")

    multitrack.save(CONST.example_dataset_path+".npz")
    tmp = pypianoroll.load(CONST.example_dataset_path+".npz")
    tmp.write(CONST.example_dataset_path+".midi")

def pianoroll2numpy(id_list):
    data = []
    # Iterate over all the songs in the ID list
    for msd_id in tqdm(id_list):
        # Load the multitrack as a pypianoroll.Multitrack instance
        song_dir = CONST.dataset_root / msd_id_to_dirs(msd_id)
        multitrack = pypianoroll.load(song_dir / os.listdir(song_dir)[0])
        # Binarize the pianorolls
        multitrack.binarize()
        # Downsample the pianorolls (shape: n_timesteps x n_pitches) #! changes the data
        multitrack.set_resolution(CONST.beat_resolution)
        # Stack the pianoroll (shape: n_tracks x n_timesteps x n_pitches)
        pianoroll = (multitrack.stack() > 0)
        # Get the target pitch range only
        pianoroll = pianoroll[:, :, CONST.lowest_pitch:CONST.lowest_pitch + CONST.n_pitches]
        # Calculate the total measures
        n_total_measures = multitrack.get_max_length() //CONST.measure_resolution
        candidate = n_total_measures - CONST.n_measures #? why? --> to avoid selecting a sample at the end of song which is not long enough
        target_n_samples = min(n_total_measures // CONST.n_measures, CONST.n_samples_per_song)
        # Randomly select a number of phrases from the multitrack pianoroll #! till here pianoroll.shape is (5 tracks, 1826 measures or beats?, 72 limited pitches) in overfitting example
        
        #! DEBUG overfit
        # for idx in np.random.choice(candidate, target_n_samples, False): #! randomly choose a measure
        start = 0
        print(f"[+] target_n_samples {target_n_samples}")
        for idx in range(target_n_samples): #! debug purpose. !!! loop modified to eliminate overlap
            end = start + CONST.n_measures * CONST.measure_resolution  # n_measures number of measures per sample
            # Skip the samples where some track(s) has too few notes
            if (pianoroll.sum(axis=(1, 2)) < 10).any():
                continue
            data.append(pianoroll[:, start:end])
            print(start,end)
            start = end
    # Stack all the collected pianoroll segments into one big array
    data = np.stack(data)
    print(f"Successfully collect {len(data)} samples from {len(id_list)} songs")
    print(f"Data shape : {data.shape} : (shape: 26154 samples from 7323 songs x  n_tracks x n_timesteps x n_pitches)")
    return data #TODO data is mutable 

def get_pianoroll_id_list():
    print("\n[+] preparing data\n")
    id_list = []
    for path in os.listdir(CONST.amg_path):
        filepath = os.path.join(CONST.amg_path, path)
        if os.path.isfile(filepath):
            with open(filepath) as f:
                id_list.extend([line.rstrip() for line in f])
    return list(set(id_list))