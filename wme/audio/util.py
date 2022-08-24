import os
from glob import glob
from scipy.io import wavfile
import h5py

def h5_to_wav(dirname,  sr_audio=125000, mic_number=1):
    fns = glob(os.path.join(dirname, '*.h5'))
    
    for i in range(len(fns)):
        with h5py.File(fns[i], 'r') as f:
            audio = f['ai_channels']['ai{}'.format(mic_number-1)][()]
        outfile = os.path.join(dirname, os.path.split(fns[i])[-1][:-3] + '_mic{}.wav'.format(mic_number))
        wavfile.write(outfile, sr_audio, audio)
        print('Wrote data to: {}'.format(outfile))
