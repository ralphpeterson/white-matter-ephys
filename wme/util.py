import numpy as np

def load_wm(bin_file):
	"""
	Load .bin file from white matter electophysiology system.

	Parameters:
	filenme: str
		Path to .bin file

	Returns:
	sr: int
		Sampling rate of physiology experiment.
	data: 2-D array
		(n_channels, n_samples) numpy array.
	"""

	#parse filename to get number of channels 
	n_channels = int(bin_file.split('_')[-2][:-2])

	#load in binary data
	_data = np.fromfile(bin_file,'int16', offset=8)

	#reshape data to (n_samples, n_samples) and scale values to MICROVOLTS
	data = _data.reshape(-1,n_channels)*6.25e3/32768

	#parse filename to get sampling rate
	sr = int(bin_file.split('_')[-1][:-7])

	return sr, data.T.astype('int16')

def get_spikes(data_phys, threshold = 5):

	"""
	Quick and dirty thresholding of phsyiology data to extract spikes.
	
	Parameters:
	data_phys: 2-D array
		Physiology data. Expects (N_channel x N_sample) array
	threshold: int
		How many standard deviations below the mean to draw the threshold
	
	Returns:
	spikes: 2-D array
		(N_channels, N_spikes) array where values in N_spikes == spike times (in units of samples)

	"""
	spikes = [] 

	for channel in range(data_phys.shape[1]):
		
		#index data from a single channels
		signal = data_phys[:, channel]

		#compute
		thresh = np.mean(signal)-(np.std(signal)*threshold)

		spikes.append(np.where(np.diff((signal < thresh).astype('int')) == -1)[0])

	return np.array(spikes, dtype=object)

def rms(signal):
	"""
	Calculate the RMS of a 1d signal. Agnostic to signal type.
	"""
	root_mean_square = np.sqrt(np.mean(signal**2))
	return root_mean_square

def test():
	return