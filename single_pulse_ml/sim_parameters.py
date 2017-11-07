import numpy as np 

class SimParams:

    def __init__(self, dm=(-0.01, 0.01), fluence=(0.1, 0.3),
                 width=(3*0.0016, 0.75), spec_ind=(-3., 3.),
                 disp_ind=2., scat_factor=(-4., -1.), 
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
        
        self._NRFI = None
        self._NSIM = None
        self.data_rfi = None
        self.y = None # FP labels 

        self._mk_plot = mk_plot
        self._NSIDE = NSIDE

    def get_false_positives(self, fn):
        f_rfi = np.load(fn)
        # Important step! Need to scramble RFI triggers. 
        np.random.shuffle(f_rfi)
        # Read in data array and labels from RFI file
        data_rfi, y = f_rfi[:, :-1], f_rfi[:, -1]

        self.data_rfi = data_rfi
        self.y = y
        self._NRFI = len(y)
        self._NSIM = len(y)

        return data_rfi, y
