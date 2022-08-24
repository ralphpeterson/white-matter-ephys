import os
import numpy as np
from scipy.signal import butter, sosfilt, sosfreqz
import shutil
import io


def chunk_bin(filename, chunk_size, channel_map): 
    
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
        
        print('Writing binary file')
        data_chanMap_reorder.astype('int16', order='F').tofile(outfile)
        
        print(data_chanMap_reorder.shape)
        print("Chunk saved to: {}".format(outfile))

        offset_counter += (chunk*2)
        print()
        
        del _data, data, data_chanMap_reorder
    
    print('Done')


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


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=0):
    """
    From https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data, axis=axis)
    return y

def preprocess(bin_files, lowcut=200, highcut=6000):
    
    for file in bin_files:
        #get sample rate from filename
        sr_phys=int(file.split('_')[-3][:-3])
        
        print('Loading data')
        data = np.fromfile(file, dtype='int16').reshape((64, -1), order='F')

        print('Bandpassing')
        data_filt = wp.butter_bandpass_filter(data, lowcut, highcut, sr_phys, axis=1)

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