import numpy as np
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter 

class CONST:
    print("[+] CONST executed by", __name__)
    import torch
    random_seed = 42 #TODO problem here is CONST file will be executed at the beggining of each iteration so torch will be randomized multiple times with the same seed
    torch.manual_seed(random_seed)
    PATH_DATASETS = './data'
    BATCH_SIZE=1 # 16
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    NUM_WORKERS=int(os.cpu_count() / 2) #! number of threats allowed
    print("[+] AVAIL_GPUS: ",AVAIL_GPUS)
    #TODO assertion for constances of piano roll
    example_dataset_path = "data/PianoRoll/overfit_results/example_data_overfit"
    dataset_path = "data/PianoRoll/lpd_5_overfit/lpd_5_cleansed/"
    dataset_root = Path(dataset_path)
    amg_path = "data/PianoRoll/amg_overfit"
    outputs_url = "data/PianoRoll/overfit_results"
    training_output_path_root = "data/PianoRoll/overfit_results/training_output_path_root"

    beat_resolution = 4  # temporal resolution of a beat (in timestep)
    measure_resolution = 4 * beat_resolution
    # Data
    n_tracks = 5  # number of tracks
    n_pitches = 72  # number of pitches
    lowest_pitch = 24  # MIDI note number of the lowest pitch
    n_samples_per_song = 1  # number of samples to extract from each song in the datset
    n_measures = 4  # number of measures per sample
    programs = [0, 0, 25, 33, 48]  # program number for each track
    is_drums = [True, False, False, False, False]  # drum indicator for each track
    track_names = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']  # name of each track
    tempo = 100
    latent_dim = 128
    n_steps = 20000
    # Sampling
    sample_interval = 1000  #! in what step interval during training we should make an example output.
    n_samples = 1 #! number of samples to be generated in the output when saving results from gan to file.  not and architecture parameter. each sample includes some measure and each measure includes some beats and notes.
    tempo_array = np.full((4 * 4 * measure_resolution, 1), tempo)

    writer = SummaryWriter("runs/summary")

    assert 24 % beat_resolution == 0, (
        "beat_resolution must be a factor of 24 (the beat resolution used in "
        "the source dataset)."
    )
    assert len(programs) == len(is_drums) and len(programs) == len(track_names), (
        "Lengths of programs, is_drums and track_names must be the same."
    )