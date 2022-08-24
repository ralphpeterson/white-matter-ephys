import numpy as np


def get_spikes(data_ephys, threshold = 4):

	"""
	Quick and dirty thresholding of phsyiology data to extract spikes.
	
	Parameters:
	data_ephys: 2-D array
		Physiology data. Expects (N_channel x N_sample) array
	threshold: int
		How many standard deviations below the mean to draw the threshold
	
	Returns:
	spikes: 2-D array
		(N_channels, N_spikes) array where values in N_spikes == spike times (in units of samples)

	"""
	spikes = [] 

	for channel in range(data_ephys.shape[1]):
		
		#index data from a single channels
		signal = data_ephys[:, channel]

		#compute
		thresh = np.mean(signal)-(np.std(signal)*threshold)

		spikes.append(np.where(np.diff((signal < thresh).astype('int')) == -1)[0])

	return np.array(spikes, dtype=object)

def rms(signal):
    rms = np.sqrt(np.mean(signal**2))
    return rms
