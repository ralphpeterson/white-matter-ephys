import h5py

def get_wm_trigger(nidaq_file):
    """
    Search the nidaq .h5 file for the trigger signal from White Matter.

    Parameters:
        nidaq_file: str, pth
    Reutrns:
        ephys_trigger: int
            wm_trigger signal (nidaq clock)
    """

    data = h5py.File(nidaq_file, 'r')
    wm_trigger = data['ephys_trigger'][0]

    return wm_trigger

#it might be good to add the get_stimuli functions from wme.audio.util here, since they depend on the nidaq file