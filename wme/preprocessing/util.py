import os
import numpy as np
import shutil
import io
from wme.util import load_wm, get_spikes, butter_bandpass_filter, reference
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import resample_poly

def chunk_bin(filename, chunk_size, channel_map, upsample=False): 
    
    """
    Split .bin file into smaller 'chunk_size' long files.

    Parameters:
    filenme: str
        Path to .bin file
    
    filenme: int
        File chunk size (minutes)
    
    channel_map: np.array
        Electrode reordering map from White Matter HS AMP 640 -> your electrode

    Returns:
        Writes chunks to 'filename' directory

    """

    #parse filename to get sampling rate
    sr = int(filename.split('_')[-1][:-7])

    #parse filename to get number of channels 
    n_channels = int(filename.split('_')[-2][:-2])

    #parse filename to get total recording duration in minutes
    recording_duration = int(filename.split('_')[-6][:-3]) + 1

    #byte offset for white matter binary header
    offset_counter = 8

    #chunk size (# elements) to read
    chunk = int(sr * 60 * chunk_size * n_channels)

    all_data = []

    for i in range(np.ceil(recording_duration/chunk_size).astype(int)):
        print('Loading chunk')
        _data = np.fromfile(filename,'int16', offset=offset_counter, count=chunk)

        #reshape data to (n_samples, n_channels) and scale values to MICROVOLTS
        print('Reshaping and converting to microvolts')
        data = _data.reshape(-1,n_channels)*6.25e3/32768
        
        basedir, fn = os.path.split(filename)
        outfile = os.path.join(basedir, fn.replace('.bin', '_{}min_chunk{}.bin'.format(chunk_size, i+1)))
                
        print('Reordering channels according to channel map')
        data_chanMap_reorder = data[:, channel_map]

        if upsample==True:
            up_factor = 4
            down_factor = 2
            sr_post = int(sr*up_factor/down_factor)
            
            data_upsampled = np.empty([int(data_chanMap_reorder.shape[0]*(up_factor/down_factor)), 64])
            for ii in range(64):
                data_ = data_chanMap_reorder[:,ii]
                # print('upsampling ch {}...'.format(ii))
                data_upsampled_ = resample_poly(data_, up_factor, down_factor)
                data_upsampled[:,ii] = data_upsampled_
            print('upsampling complete')
            data_chanMap_reorder = data_upsampled
        
        print('Writing binary file')
        data_chanMap_reorder.astype('int16', order='F').tofile(outfile)
        
        print(data_chanMap_reorder.shape)
        print("Chunk saved to: {}".format(outfile))

        offset_counter += (chunk*2)
        print()
        
        del _data, data, data_chanMap_reorder
    
    print('Done')

def preprocess(bin_files, lowcut=200, highcut=6000):
    
    for file in bin_files:
        #get sample rate from filename
        sr_phys=int(file.split('_')[-3][:-3])
        
        print('Loading data')
        data = np.fromfile(file, dtype='int16').reshape((64, -1), order='F')

        print('Bandpassing')
        data_filt = butter_bandpass_filter(data, lowcut, highcut, sr_phys, axis=1)

        print('Computing common reference')
        cmr = np.median(data_filt, axis=0) #common median reference

        print('Subtracting reference from bandbpassed data')
        data_filt_referenced = data_filt - np.tile(cmr, (data_filt.shape[0], 1))
        
        print('Saving pre-processed data: {}'.format(file.replace('.bin', '_preprocessed.bin')))
        data_filt_referenced.astype('int16', order='F').T.tofile(file.replace('.bin', '_preprocessed.bin'))
        print()
    print('Done')

def merge_bins(bin_files, outfile):

    """
    Merge .bin files into single file.

    Parameters:
    bin_files: list
        File paths to all the files you want to merge. Make sure you sort them in the right order...
    
    outfile: path, str
        Merge file name.

    """

    with open(outfile,'wb') as dest:
        for file in bin_files:
            with open(file,'rb') as f:
                shutil.copyfileobj(f, dest, length=io.DEFAULT_BUFFER_SIZE)
    print('Merge file saved to: {}'.format(outfile))

def save_waveforms(data_phys, sr, spikes, n_waveforms=200, bin_file='', spike_threshold=5):
    """
    Function to save PNGs of spike waveforms detected on different channels.
    """
    for ich in range(data_phys.shape[0]):
        plt.figure()
        all_traces = []
        for ispike in range(1, len(spikes[ich])-1)[:n_waveforms]:
            working_start = int(spikes[ich][ispike] - sr*.001)
            working_stop = int(spikes[ich][ispike] + sr*.001)
            working_trace = data_phys[ich, working_start:working_stop]
            all_traces.append(working_trace)
            plt.plot(working_trace, 'gray', alpha=0.25)
        
        plt.plot(np.mean(np.array(all_traces), axis=0), 'k')
        plt.ylabel('microvolts')
        plt.xlabel('Time (ms)')
        plt.ylim(-200, 200) #TODO: auto-delete outliers so that the y axis doesn't blow out of proportion
        plt.xticks(np.arange(0, sr*.001*2, sr*.001*2/6),
         np.around(np.arange(0, sr*.001*2, sr*.001*2/6)/sr*1000, decimals=1))
        plt.title('Thresholded spike detected on ch {}'.format(ich))
        sns.despine()
        plt.tight_layout()
        
        dirname, basename = os.path.split(bin_file)
        outdir = os.path.join(dirname, 'spike_waveforms_th{}'.format(spike_threshold))
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outfile = os.path.join(outdir, 'channel_{}'.format(ich))
        plt.savefig(outfile)
        plt.close()

def check_waveforms(bin_file, phys_bandpass=(200,6000), n_waveforms=200, spike_threshold=5, save_spikes=True):
    """
    A function to grab thresholded spikes on each channel and overlay spike waveforms to check for signal.
    """

    #load raw data
    print('Loading data')
    sr, data = load_wm(bin_file)

    #bandpass
    print('Bandpassing between {}-{} Hz'.format(phys_bandpass[0], phys_bandpass[1]))
    data_filt = butter_bandpass_filter(data, phys_bandpass[0], phys_bandpass[1], sr, axis=1)
    
    #subtract off reference
    print('Computing and subtracting off common reference')
    data_phys = reference(data_filt)

    #get spikes
    print('Getting spikes')
    spikes = get_spikes(data_phys, threshold=spike_threshold, save_spikes=save_spikes, outdir=os.path.split(bin_file)[0])

    #save spikes
    print('Saving spike waveforms PNGs')
    save_waveforms(data_phys, sr, spikes, n_waveforms=n_waveforms, bin_file=bin_file, spike_threshold=spike_threshold)
