import numpy as np
import os
from pathlib import Path

class CONST:
    print("[+] CONST executed by", __name__)

    import torch
    random_seed = 42 #TODO problem here is CONST file will be executed at the beggining of each iteration so torch will be randomized multiple times with the same seed
    torch.manual_seed(random_seed)

    dataset_root = './data/PianoRoll/dataset'
    example_dataset_path = "data/PianoRoll/results/genre"
    dataset_path = dataset_root+"/lpd_5/lpd_5_cleansed/"
    amg_path = dataset_root+"/amg"
    outputs_url = "data/PianoRoll/results/genre"
    training_output_path_root = "data/PianoRoll/results/genre/training_output_path_root"

    BATCH_SIZE=64
    sample_interval = 200  #! in what step interval during training we should make an example output.
    # n_steps = 20000
    n_steps = 10000
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    print("[+] AVAIL_GPUS: ",AVAIL_GPUS)
    # NUM_WORKERS=int(os.cpu_count() / 2) #! number of threats allowed
    #TODO assertion for constances of piano roll

    beat_resolution = 4  # temporal resolution of a beat (in timestep)
    measure_resolution = 4 * beat_resolution
    # Data
    n_tracks = 2 # 5  # number of tracks
    n_pitches = 72  # number of pitches
    lowest_pitch = 24  # MIDI note number of the lowest pitch
    n_samples_per_song = 5  # number of samples to extract from each song in the datset
    n_measures = 4  # number of measures per sample
    programs = [0, 0, 25, 33, 48]  # program number for each track
    is_drums = [True, False, False, False, False]  # drum indicator for each track
    os_bass = [False, False, False, True, False]
    track_names = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']  # name of each track
    tempo = 100
    latent_dim = 128
    # Sampling
    n_samples = 10 #! number of samples to be generated in the output when saving results from gan to file.  not and architecture parameter. each sample includes some measure and each measure includes some beats and notes.
    tempo_array = np.full((4 * 4 * measure_resolution, 1), tempo)
    assert 24 % beat_resolution == 0, (
        "beat_resolution must be a factor of 24 (the beat resolution used in "
        "the source dataset)."
    )
    assert len(programs) == len(is_drums) and len(programs) == len(track_names), (
        "Lengths of programs, is_drums and track_names must be the same."
    )

    genre_code = {
        'Folk' : 0 ,
        'Country' : 1 ,
        'Rap' : 2 ,
        'Blues' : 3 ,
        'RnB' : 4 ,
        'New-Age' : 5 ,
        'Vocal' : 6 ,
        'Reggae' : 7 ,
        'Pop_Rock' : 8 ,
        'Electronic' : 9 ,
        'International' : 10 ,
        'Jazz'  : 11 ,
        'Latin' : 12 ,
    }