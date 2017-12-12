# training data 30 September 2017
# miscellaneous tools for preparing and processing 
# machine learning data 

import numpy as np
import glob

import dataproc

def save_background_data(fdir, outfile=None):
    """ Read in randomly selected Pathfinder data in directory fdir,
    dedisperse to a DM between 25 and 2000 pc cm**-3,
    and create a large array of (nfreq, ntime_pulse) arrays 
    over which FRBs can be injected. 
    These data haven't been RFI cleaned! Could cause problems.
    """
    fl = glob.glob(fdir)
    fl.sort()
    arr_full = []

    nfreq = 16
    freq_rebin = 2.0
    ntime_pulse = 250

    for ff in fl[:75]:
        arr = np.load(ff)[:, 0]
        arr[arr!=arr] = 0.
        nfreq_arr, ntime = arr.shape

        # Disperse data to random dm
        _dm = np.random.uniform(25, 2000.0)
        arr = dedisperse_data(arr, _dm, freq=np.linspace(800, 400, nfreq_arr))

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
    arr_full = arr_full.reshape(-1, nfreq*ntime_pulse)

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

def calc_snr(arr, ntime=250):
    """ Calculate the S/N of pulse profile after 
    trying 4 rebinnings.

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

    snr = 0
    for ii in range(1, 5):
        arr = arr[:len(arr)//ii*ii].reshape(-1, ii).mean(-1)
        snr = max(snr, arr.max() / np.std(arr[:ntime//ii//4]))

    return snr
