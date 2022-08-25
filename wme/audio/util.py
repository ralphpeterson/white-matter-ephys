import os
from glob import glob
from scipy.io import wavfile
from wme.nidaq.util import get_wm_trigger


def h5_to_wav(dirname,  sr_audio=125000, mic_number=1):
    fns = glob(os.path.join(dirname, '*.h5'))
    
    for i in range(len(fns)):
        with h5py.File(fns[i], 'r') as f:
            audio = f['ai_channels']['ai{}'.format(mic_number-1)][()]
        outfile = os.path.join(dirname, os.path.split(fns[i])[-1][:-3] + '_mic{}.wav'.format(mic_number))
        wavfile.write(outfile, sr_audio, audio)
        print('Wrote data to: {}'.format(outfile))

def truncate_audio(nidaq_file, data_phys, combine_audio = False):

    #this is kinda a dumb function

	"""
	Truncate beginning and end of audio to be aligned with physiology data and same length (in seconds)
	
	Parameters:
	exp_dir: str
		directory where phys and analog data live

	data_phys: 2-D array
		output from load_wm() - a 2-D array that is (n_samples x n_channels)

	combine_audio: boolean
		if True, function ouputs a single 1-D audio array with signal average from all mics.
		if False, function outputs N 1-D audio arrays for each mic. 

	Returns:
	audio: N-D array

	if combine_audio True:
		audio: 1-D array:
			average audio signal

	if combine_audio False:
		mic1, mic2: 2-D array:
			audio signals

	"""

	

	# ephys_trigger = get_wm_trigger(nidaq_file)
    
    # print('Ephys trigger detected at analog sample number {}'.format(ephys_trigger))
	# print()

	# #calculate the end truncation sample number
	# #TODO: don't hard code the analog/ephys sampling ratio
	# sampling_rate_ratio = 10
	# end_trunc = (len(data_phys)*sampling_rate_ratio)

	# if combine_audio == True:
	# 	_mic1 = data_analog['analog_input'][0]
	# 	mic1 = _mic1[ephys_trigger_rising_edge:ephys_trigger_rising_edge+end_trunc]

	# 	_mic2 = data_analog['analog_input'][1]
	# 	mic2 = _mic2[ephys_trigger_rising_edge:ephys_trigger_rising_edge+end_trunc]

	# 	audio = np.mean(np.array([mic1, mic2]), axis=0)

	# 	return np.array([audio]), ephys_trigger_rising_edge
	
	# else:
	# 	_mic1 = data_analog['analog_input'][0]
	# 	mic1 = _mic1[ephys_trigger_rising_edge:ephys_trigger_rising_edge+end_trunc]

	# 	_mic2 = data_analog['analog_input'][1]
	# 	mic2 = _mic2[ephys_trigger_rising_edge:ephys_trigger_rising_edge+end_trunc]

	# 	return np.array([mic1, mic2]), ephys_trigger_rising_edge