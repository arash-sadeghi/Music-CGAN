import pypianoroll
from CONST_VARS import CONST 
import os
def midi_extractor(npz_url):
    multitrack_in = pypianoroll.load(npz_url)

    tracks =[]
    for track in multitrack_in.tracks:
        if track.name == 'Bass' or track.name == 'Drums':
            tracks.append(track)

    multitrack_out = pypianoroll.Multitrack(tracks=tracks, tempo =multitrack_in.tempo )

    # multitrack_out.write(os.path.join(''))
    multitrack_out.write('data\PianoRoll\dataset\sample_rock_song_from_dataset_DB.midi')
    multitrack_in.write('data\PianoRoll\dataset\sample_rock_song_from_dataset_all_instruments__.midi')

    print("done")

if __name__ == '__main__':
    midi_extractor('data\PianoRoll\dataset\lpd_5\lpd_5_cleansed\W\M\H\TRWMHMP128EF34293F\d8392424ea57a0fe6f65447680924d37.npz')