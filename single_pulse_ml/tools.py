# training data 30 September 2017
# miscellaneous tools for preparing and processing 
# machine learning data 

import numpy as np
import glob
import scipy.signal

import dataproc

def save_background_data(fdir, outfile=None, nfreq = 32):
    """ Read in randomly selected Pathfinder data in directory fdir,
    dedisperse to a DM between 25 and 2000 pc cm**-3,
    and create a large array of (nfreq, ntime_pulse) arrays 
    over which FRBs can be injected. 
    These data haven't been RFI cleaned! Could cause problems.
    """
    fl = glob.glob(fdir)
    fl.sort()
    arr_full = []

    freq_rebin = 1
    ntime_pulse = 250

    for ff in fl[:75]:
        print(ff)
        arr = np.load(ff)[:, 0]
        arr[arr!=arr] = 0.
        nfreq_arr, ntime = arr.shape
        print(arr.shape)

        # Disperse data to random dm
        _dm = np.random.uniform(25, 2000.0)
        arr = dedisperse_data(arr, _dm)

        # rebin to nfreq, divide data into blocks of len ntime_pulse
        arr = np.nansum(arr.reshape(-1, freq_rebin, ntime), axis=1)/freq_rebin
        arr = arr[:, :ntime//ntime_pulse*ntime_pulse]
        arr = arr.reshape(nfreq, -1, ntime_pulse)
        arr_full.append(arr)

    # Reorganize array to be (ntriggers, nfreq, ntime_pulse)
    arr_full = np.concatenate(arr_full)[:, :ntime//ntime_pulse*ntime_pulse]
    arr_full = arr_full.reshape(-1, nfreq, ntime//ntime_pulse, ntime_pulse)
    arr_full = np.transpose(arr_full, (0, 2, 1, 3)).reshape(-1, nfreq, ntime_pulse)

    # Go through each noise trigger and add data 
    for ii, arr in enumerate(arr_full):
        arr_full[ii] = dataproc.normalize_data(arr)

    # Reshape to have same shape as RFI triggers
    #arr_full = arr_full.reshape(-1, nfreq*ntime_pulse)
    np.random.shuffle(arr_full)

    if outfile is not None:
        np.save(outfile, arr_full)

    return arr_full

def dedisperse_data(f, _dm, freq_bounds=(800,400), dt=0.0016, freq_ref=600):
    """ Dedisperse data to some dispersion measure _dm.
    Frequency is in MHz, dt delta time in seconds.
    f is data to be dedispersed, shaped (nfreq, ntime)
    """

    # Calculate the number of bins to shift for each freq
    NFREQ=f.shape[0]
    freq = np.linspace(freq_bounds[0], freq_bounds[1], NFREQ)
    ind_delay = ((4.148808e3 * _dm * (freq**(-2.) - freq_ref**(-2.))) / dt).astype(int)
    for ii, nu in enumerate(freq):
        f[ii] = np.roll(f[ii], -ind_delay[ii])

    return f

def calc_snr(arr, fast=False):
    """ Calculate the S/N of pulse profile after 
    trying 9 rebinnings.

    Parameters
    ----------
    arr   : np.array
        (ntime,) vector of pulse profile 
    ntime : np.int 
        number of times in profile

    Returns
    -------
    snr : np.float 
        S/N of pulse
    """
    assert len(arr.shape)==1
    
    ntime = len(arr)
    snr_max = 0
    widths = [1, 2, 4, 8, 16, 32, 64, 128]

#    for ii in range(1, 10):
    for ii in widths:

        # skip if boxcar width is greater than 1/4th ntime
        if ii > ntime//8:
            continue
            
        arr_copy = arr.copy()
        arr_ = arr_copy[:len(arr)//ii*ii].reshape(-1, ii).mean(-1)

        if fast is False:
            std_chunk = scipy.signal.detrend(arr_, type='linear')
            std_chunk.sort()
            ntime_r = len(std_chunk)
            stds = 1.148*np.sqrt((std_chunk[ntime_r/40:-ntime_r/40]**2.0).sum() /
                                    (0.95*ntime_r))
            snr_ = std_chunk[-1] / stds 
        else:
            sig = np.std(arr_[:len(arr_)//3])
            snr_ =  arr_.max() / sig
        
        if snr_ > snr_max:
            snr_max = snr_
            width_max = ii

    return snr_max








