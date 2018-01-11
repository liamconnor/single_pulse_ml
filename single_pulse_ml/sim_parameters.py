import os
import time

import numpy as np 
import h5py

class SimParams:

    def __init__(self, dm=(-0.01, 0.01), fluence=(0.1, 0.3),
                 width=(3*0.0016, 0.75), spec_ind=(-3., 3.),
                 disp_ind=2., scat_factor=(-4., -1.), NRFI=None, NSIM=None,
                 SNR_MIN=10., SNR_MAX=100., out_file_name=None, 
                 NTIME=250, NFREQ=16, mk_plot=False, NSIDE=8):

        self._dm = dm
        self._fluence = fluence 
        self._width = width 
        self._spec_ind = spec_ind
        self._disp_ind = disp_ind
        self._scat_factor = scat_factor

        self._SNR_MIN = SNR_MIN
        self._SNR_MAX = SNR_MAX
        self._NTIME = NTIME
        self._NFREQ = NFREQ
        self._out_file_name = out_file_name
        
        self._NRFI = NRFI
        self._NSIM = NSIM
        self.data_rfi = None
        self.y = None # FP labels 

        self._mk_plot = mk_plot
        self._NSIDE = NSIDE

    def generate_noise(self):
        y = np.zeros([self._NRFI])
        noise = np.random.normal(0, 1, self._NRFI*self._NTIME*self._NFREQ)
        noise = noise.reshape(-1, self._NFREQ*self._NTIME)
        self._NSIM = self._NRFI

        return noise, y

    def get_false_positives(self, fn):

        ftype = fn.split('.')[-1]

        if ftype in ('hdf5', 'h5'):
            f = h5py.File(fn)
            data_rfi = f['data_freq_time'][:]
            data_rfi = data_rfi.reshape(len(data_rfi), -1)
            y = f['labels'][:]
        elif ftype in ('npy',):
            f_rfi = np.load(fn)
            # Important step! Need to scramble RFI triggers. 
            np.random.shuffle(f_rfi)
            # Read in data array and labels from RFI file
            data_rfi, y = f_rfi[:, :-1], f_rfi[:, -1]
        else:
            return 
    
        if self._NRFI is not None:
            if self._NSIM is None:
                self._NSIM = self._NRFI
                
            self.data_rfi = data_rfi[:self._NRFI]
            self.y = y[:self._NRFI]
        else:
            self._NRFI = len(y)
            self._NSIM = self._NRFI 
            self.data_rfi = data_rfi[:self._NSIM]
            self.y = y[:self._NSIM]

        return data_rfi, y

    def write_sim_data(self, data_freq_time, labels, fnout, 
                       data_dm_time=None, params=None, snr=None):

        ftype = fnout.split('.')[-1]

        if os.path.exists(fnout):
            t0_str = time.strftime("_%Y_%m_%d_%H:%M:%S", time.gmtime())
            fnout = fnout.split(ftype)[0][:-1] + t0_str + '.' + ftype

        if ftype in ('hdf5', 'h5'):

            f = h5py.File(fnout)
            f.create_dataset('data_freq_time', data=data_freq_time)
            f.create_dataset('labels', data=labels)

            if data_dm_time is not None:
                f.create_dataset('data_dm_time', data=data_dm_time)
            if params is not None:
                f.create_dataset('params', data=params)
            if snr is not None:
                f.create_dataset('snr', data=snr)

            f.close()

        elif ftype in ('npy'):
            np.save(fnout, data)







