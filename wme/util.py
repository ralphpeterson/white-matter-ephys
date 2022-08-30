import numpy as np
from scipy.signal import butter, sosfilt, sosfreqz
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from wme.nidaq.util import get_wm_trigger
from wme.audio.util import get_stimuli

sns.set_style('white')
sns.set_style('ticks')
sns.set_context('talk')

def butter_bandpass(lowcut, highcut, fs, order=5):
	"""
	From https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html

	Might be a better way to implement here that uses second-order sections: 
	https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
	"""
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	sos = butter(order, [low, high], analog=False, btype='band', output='sos')
	return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=1):
	"""
	From https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
	"""
	sos = butter_bandpass(lowcut, highcut, fs, order=order)
	y = sosfilt(sos, data, axis=axis)
	return y

def cmr(data_phys_bandpassed):
	"""
	Compute the common median reference for an n_channel x n_sample array of ephys data.
	
	Returns:
		Common reference for all channels.
	"""
	
	cmr = np.median(data_phys_bandpassed, axis=0)
	
	return cmr

def reference(data_phys_bandpassed):
	"""
	Subtract off the common reference (common median reference) from your bandpasses data.
	"""
	ref = cmr(data_phys_bandpassed)
	data_phys = data_phys_bandpassed - ref.reshape((1, -1))

	return data_phys

def load_wm(bin_file, phys_bandpass=False):
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

	print('Loading data')
	#load in binary data
	_data = np.fromfile(bin_file,'int16', offset=8)

	#reshape data to (n_samples, n_samples) and scale values to MICROVOLTS
	data = _data.reshape(-1,n_channels)*6.25e3/32768
	data = data.T.astype('int16') #transpose and convert back to int 16
	
	#parse filename to get sampling rate
	sr = int(bin_file.split('_')[-1][:-7])

	if phys_bandpass == False:
		return sr, data
	else:
		#bandpass
		print('Bandpassing between {}-{} Hz'.format(phys_bandpass[0], phys_bandpass[1]))
		data_filt = butter_bandpass_filter(data, phys_bandpass[0], phys_bandpass[1], sr, axis=1)

		#subtract off reference
		print('Computing and subtracting off common reference')
		data_phys = reference(data_filt)

		return sr, data_phys

def get_spikes(data_phys, threshold = 5, save_spikes=True, outdir=''):

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

	for channel in range(data_phys.shape[0]):
		
		#index data from a single channels
		signal = data_phys[channel, :]

		#compute
		thresh = np.mean(signal)-(np.std(signal)*threshold)

		spikes.append(np.where(np.diff((signal < thresh).astype('int')) == -1)[0])
	
	if save_spikes == True:
		outfile = os.path.join(outdir,'spikes_th{}.npy'.format(threshold))
		print('Spikes saving to {}'.format(outfile))
		np.save(outfile, np.array(spikes, dtype=object))
		return np.array(spikes, dtype=object)
	else:
		return np.array(spikes, dtype=object)

def rms(signal):
	"""
	Calculate the RMS of a 1d signal. Agnostic to signal type.
	"""
	root_mean_square = np.sqrt(np.mean(signal**2))
	return root_mean_square

def truncate_spikes(spikes, onset, offset, ephys_trigger, sr_nidaq=125000, sr_phys=25000):
	"""
	Returns: 
	Spikes truncated between the onset/offset input. Converts from nidaq to WM clock.
	"""
	
	nidaq_oo = (np.array([onset, offset])-ephys_trigger)/sr_nidaq #time in s that onset/offset occurs
	phys_oo = (nidaq_oo*sr_phys).astype('int') #convert the onset/offset back to phys idx

	spikes_trunc = []
	#loop through spikes on all channels and extract spikes for this time window
	for i in range(len(spikes)):
		working_spikes_trunc = np.array([s for s in spikes[i] if phys_oo[0] <= s <= phys_oo[1]],dtype=object)-phys_oo[0]
		spikes_trunc.append(working_spikes_trunc)
	
	return spikes_trunc

def get_stim_spikes(stimulus, onsets, spikes, ephys_trigger,
					pad=1, sr_phys=25000, sr_nidaq=125000):
	"""
	Get all the spikes that occur during the stimulus windows.
	Parameters:
		Onsets: np.array
			Onsets of audio stimuli. Nidaq clock.
		pad: time in s pre/post stimulus to include.
	"""
	all_spikes = []
	for onset in onsets:
		dur = len(stimulus)
		window_onset = onset - (pad*sr_nidaq)
		window_offset = onset + dur + (pad*sr_nidaq)

		spikes_trunc = truncate_spikes(spikes,
						onset = window_onset,
						offset = window_offset,
						ephys_trigger=ephys_trigger,
						sr_nidaq=sr_nidaq,
						sr_phys=sr_phys)

		all_spikes.append(spikes_trunc)
	
	return np.array(all_spikes, dtype=object)

def plot_psth_channel(all_spikes, data_phys, stimulus, sr_nidaq=125000,
					  sr_phys=25000, pad=1, hist_binsize=0.1, outdir=''):
	
	window_size = len(stimulus) + (2*pad*sr_nidaq)
	pad = pad*sr_nidaq #convert to nidaq clock
	
	for i in range(data_phys.shape[0]):
		#get all spikes over all stims for a given channel
		ch = np.vstack(all_spikes)[:,i]

		plt.figure(figsize=(12,10))

		plt.subplot(411)
		plt.specgram(stimulus, NFFT=512, noverlap=256, Fs=sr_nidaq, 
				 xextent=(pad,window_size-pad), cmap='magma');
		plt.axis('off')
		plt.xlim(0, window_size)


		plt.subplot(412)
		plt.plot(np.arange(pad, window_size-pad), stimulus, 'k')
		plt.axis('off')
		plt.xlim(0, window_size)
		sns.despine(bottom=True, right=True);    

		plt.subplot(413)
		n, bins, patches = plt.hist(np.hstack(ch), range=(0, window_size/sr_nidaq*sr_phys),
									bins=int(window_size/(hist_binsize*sr_nidaq)), 
									histtype='step', color='k')

		plt.xticks([])
		plt.ylabel('counts \n ({} s bin)'.format(hist_binsize), rotation=0, labelpad=40, fontsize=14)
		plt.xlim(0, window_size/sr_nidaq*sr_phys)
		sns.despine(bottom=True, right=True);

		plt.subplot(414)
		for j in range(len(ch)):
			plt.plot(ch[j], [j]*len(ch[j]), '|k')

		plt.xticks(np.arange(0, window_size/sr_nidaq*sr_phys, int(sr_phys*.5)), 
				   np.arange(0, window_size/sr_nidaq*sr_phys, int(sr_phys*.5))/sr_phys)

		plt.xlabel('time (s)')
		plt.ylabel('sound trial', rotation=0, labelpad=40, fontsize=14)
		plt.xlim(0, window_size/sr_nidaq*sr_phys)
		sns.despine()
		
		plt.title('Channel {}'.format(i+1))
		plt.savefig(os.path.join(outdir, 'channel{}.png'.format(i)), dpi=300, transparent=False)
		plt.close()

def get_psth_traces(all_spikes, stimulus, data_phys, sr_nidaq=125000, sr_phys=25000, 
					pad=1, hist_binsize=0.1):
	window_size = len(stimulus) + (2*pad*sr_nidaq)
	pad = pad*sr_nidaq #convert to nidaq clock
	
	#hold on to the histogram traces to plot later
	psth_traces = []
	for i in range(data_phys.shape[0]):
		#get all spikes over all stims for a given channel
		ch = np.vstack(all_spikes)[:,i]
		
		n, bins, patches = plt.hist(np.hstack(ch), range=(0, window_size/sr_nidaq*sr_phys),
									bins=int(window_size/(hist_binsize*sr_nidaq)), 
									histtype='step', color='k')
		plt.close()
		psth_traces.append(np.array([n, bins], dtype=object))
	return psth_traces

def psth_channel(exp_dir, phys_bandpass=(200, 6000), spike_threshold=5, pad=1, save_psth_traces=True):
	
	#define paths to relevant data
	bin_file = glob(os.path.join(exp_dir, '*.bin'))[0]
	nidaq_file = glob(os.path.join(exp_dir, '*.h5'))[0]
	stimulus_folder = os.path.join(exp_dir, 'stimuli')

	if np.alltrue([os.path.exists(i) for i in [bin_file, nidaq_file, stimulus_folder]]):
		pass
	else:
		print('Check that bin_file, nidaq_file, and stimnulus_folder exist.')
		return
	
	#load physiology data
	sr_phys, data_phys = load_wm(bin_file, phys_bandpass=phys_bandpass)

	#get spikes
	spikes_file = glob(os.path.join(exp_dir, '*spikes_th'))
	if len(spikes_file) == 0:
		print('Getting spikes')
		spikes = get_spikes(data_phys, threshold=spike_threshold, save_spikes=True, outdir=exp_dir)
	else:
		print('Loading spikes from {}'.format(spikes_file[0]))
		spikes = np.load(spikes_file[0])

	#get stim times
	print('Getting stimulus times')
	d, stimuli, sr_nidaq = get_stimuli(nidaq_file, stimulus_folder)
	
	
	for istim in range(len(d.keys())):
		working_stim_name = list(d.keys())[istim]
		print('Computing PSTHs for {}'.format(working_stim_name))   
		#aggregate all the spikes in a list
		all_spikes = get_stim_spikes(stimuli[istim], d[working_stim_name], spikes, 
									 get_wm_trigger(nidaq_file),
									 pad=pad, sr_phys=sr_phys, sr_nidaq=sr_nidaq)
		
		outdir = os.path.join(exp_dir, 'psth_th{}_{}'.format(spike_threshold, working_stim_name))
		if not os.path.exists(outdir):
			os.makedirs(outdir)
			
		plot_psth_channel(all_spikes, data_phys, stimuli[istim], sr_nidaq=sr_nidaq, 
						  sr_phys=sr_phys, pad=pad, hist_binsize=0.1, outdir=outdir)
		
		if save_psth_traces == True:
			outfile = os.path.join(outdir, 'psth_{}_{}.npy'.format(spike_threshold, working_stim_name))
			print('Computing and saving PSTH traces to {}'.format(outfile))
			psth_traces = get_psth_traces(all_spikes, stimuli[istim], data_phys, 
										  sr_nidaq=sr_nidaq, sr_phys=sr_phys, 
										  pad=pad, hist_binsize=0.1)
			np.save(outfile, psth_traces)
		else:
			return

def test():
	return
	
