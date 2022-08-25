import os
from glob import glob
from scipy.io import wavfile
from wme.nidaq.util import get_wm_trigger
import h5py
import numpy as np


def h5_to_wav(dirname, sr_audio=125000, mic_number=1):
	"""
	Convert audio stored in .h5 file to wav file.
	"""
	fns = glob(os.path.join(dirname, '*.h5'))
	for i in range(len(fns)):
		with h5py.File(fns[i], 'r') as f:
			audio = f['ai_channels']['ai{}'.format(mic_number-1)][()]
		outfile = os.path.join(dirname, os.path.split(fns[i])[-1][:-3] + '_mic{}.wav'.format(mic_number))
		wavfile.write(outfile, sr_audio, audio)
		print('Wrote data to: {}'.format(outfile))

def get_audio(stimulus_folder):
	"""
	Get audio stimuli and their names for a given experiment. Expects a folder called "stimuli" containing .wav files of all stimuli used in your experiment.
	"""
    
	fns = np.sort(glob(os.path.join(stimulus_folder, '*.wav')))
	stimuli = []
	stimulus_names = []
    
	for fn in fns:
		sr, audio = wavfile.read(fn)
		stimuli.append(audio)
		stimulus_names.append(os.path.basename(fn)[:-4])
        
	return stimuli, stimulus_names

def get_stimuli(nidaq_file, stimulus_folder):
	"""
	Creates a dictionary of stim names/times and a list of raw audio for each stimuli.
	"""

    #load nidaq file
	data = h5py.File(nidaq_file, 'r')
    #get the nidaq timestamps for the audio stimuli
	stim_times = data['audio_onset'][:]    
    #get the audio stimuli and their names
	stimuli, stimulus_names = get_audio(stimulus_folder)
    #create a dictionary to store the stim names and times
	d = {}
	stim_ttl = np.unique(stim_times[:,1])
	for i in range(len(stim_ttl)):
		d[stimulus_names[i]] = stim_times[stim_times[:,1] == stim_ttl[i],0]
    
	return d, stimuli