import time

import numpy as np
import matplotlib.pylab as plt
from astropy.time import Time, TimeDelta

"""
Need to replace preproc cleaning method with 
Dany's fuller method.plt.

maybe give DadaHeader an option for stokes trigger or not
"""


class DadaHeader:
    """ Class to generate header object with 
    information about current buffer data. 
    If trigger==True, information about the event 
    (dm, beamno, etc.) will be in the header.
    """

    def __init__(self, header, trigger=False):
        self.dt = np.float(header['TSAMP'])
        self.nchan = np.int(header['NCHAN'])
        self.freq_mid = np.float(header['FREQ']) # mid frequency?
        self.freq_low = np.float(header['MIN_FREQUENCY'])
        self.dnu = np.float(header['CHANNEL_BANDWIDTH'])
        self.bw = np.float(header['BW'])
        self.freq_high = self.freq_low + self.nchan*self.dnu
        self.RA = np.float(header['RA'])
        self.DEC = np.float(header['DEC'])
        self.file_number = np.int(header['FILE_NUMBER'])
        self.ntime_batch = np.int(header['SAMPLES_PER_BATCH'])
        self.beamno = np.int(header['BEAM'])
        self.bytes_per_second = np.float(header['BYTES_PER_SECOND'])

        # the following three are start times of observation
        self.utc_start = header['UTC_START']
        self.lst_start = header['LST_START']
        self.mjd_start = np.float(header['MJD_START'])
        self.astropy_start_time = Time(self.mjd_start, format='mjd', scale='utc')

        # start byte of current page
        self.obs_offset_bytes = np.float(header['OBS_OFFSET'])
        self.obs_offset_seconds = self.obs_offset_bytes/self.bytes_per_second

        # start time of current page
        self.astropy_page_time = self.astropy_start_time + TimeDelta(self.obs_offset_seconds, format='sec')

        if trigger:
            self.dm = np.float(header['EVENT_DM'])
            self.width = np.float(header['EVENT_WIDTH'])
            self.snr = np.float(header['EVENT_SNR'])
            self.beamno = np.int(header['EVENT_BEAM'])
        else:
            self.dm = None
            self.width = None
            self.snr = None

class RealtimeProc:

    def __init__(self, dt=8.192e-5):
        self.dt = dt

    def cleandata(self, data, threshold=3.0, dumbmask=True):
        """ Take filterbank object and mask 
        RFI time samples with average spectrum.

        Parameters:
        ----------
        data : np.ndarray
            (nfreq, ntime) array
        threshold : float 
            units of sigma

        Returns:
        -------
        cleaned filterbank object
        """
    #    logging.info("Cleaning RFI")
        if dumbmask:
            try:
                dumb_rfimask = np.loadtxt('/home/arts/ARTS-obs/amber_conf/zapped_channels.conf')
                dumb_rfimask = list(dumb_rfimask.astype(int))
            except:
                dumb_rfimask = []
        else:
            dumb_rfimask = []

        assert len(data.shape)==2, "Expected (nfreq, ntime) array"

        data[dumb_rfimask] = 0.

        dtmean = np.mean(data, axis=-1)
        dfmean = np.mean(data, axis=0)

        stdevf = np.std(dfmean)
        medf = np.median(dfmean)
        maskf = list(np.where(np.abs(dfmean - medf) > threshold*stdevf)[0])

        maskf = dumb_rfimask + maskf

        # replace with mean spectrum
        data[:, maskf] = dtmean[:, None]*np.ones(len(maskf))[None]

        return data

    def dedisperse(self, data, dm, freq=(1550, 1250), freq_ref=None):
        nfreq, ntime = data.shape[0], data.shape[1]

        freqs = np.linspace(freq[0], freq[-1], nfreq)

        if freq_ref is None:
            freq_ref = freqs.max()

        tdelay = 4.148e3*dm*(freqs**-2 - freq_ref**-2)
        ntime = len(data[0])

        maxind_arr = []

        for ii, f in enumerate(freqs):
            data[ii] = np.roll(data[ii], -np.int(tdelay[ii]/self.dt))

        return data

    def preprocess(self, data, invert_spectrum=False, threshold=3.0):
        if len(data.shape)==2:
            data = data[None]

        if invert_spectrum:
            data = data[:, ::-1]

        ntab = data.shape[0]

        for tab in range(ntab):
            data[tab] = self.cleandata(data[tab], threshold=threshold)

        if data.shape[0]==1:
            data = data[0]

        return data 

    def dedisperse_tabs(self, data, dm, dt=8.192e-5, freq=(1550, 1250), freq_ref=None):

        if len(data.shape)==2:
            data = data[None]

        ntab = data.shape[0]

        for tab in range(ntab):
            data[tab] = self.dedisperse(data[tab], dm, freq=freq, freq_ref=freq_ref)

        return data

    def postprocess(self, data, nfreq_plot=32, ntime_plot=64, downsample=1):
        if len(data.shape)==2:
            data = data[None]

        nfreq = data.shape[1]
        ntab = data.shape[0]
        ntime = data.shape[-1]
        ntime_ds = ntime//downsample
        data_classify = np.empty([ntab, nfreq_plot, ntime_plot])

        for tab in range(ntab):
            data_tab = data[tab]
            data_tab = data_tab.reshape(nfreq_plot, nfreq//nfreq_plot, -1).mean(1)
            data_tab = data_tab[:, :ntime//downsample*downsample]
            data_tab = data_tab.reshape(-1, ntime//downsample, downsample).mean(-1)

            maxind = np.argmax(data_tab.mean(0))

            if (ntime_ds-maxind)<ntime_plot//2:
                maxind = ntime_ds - ntime_plot//2
            if maxind<ntime_plot//2:
                maxind = ntime_plot//2

            data_tab -= np.median(data_tab)
            data_tab /= np.std(data_tab)
            data_tab[data_tab!=data_tab] = 0.
#            print("ntime, maxind, data_tab[:, maxind-ntime_plot//2:maxind+ntime_plot//2].shape, data_classify.shape, data_tab.shape")
#            print(ntime, maxind, data_tab[:, maxind-ntime_plot//2:maxind+ntime_plot//2].shape, data_classify.shape, data_tab.shape)
            data_classify[tab] = data_tab[:, maxind-ntime_plot//2:maxind+ntime_plot//2]

        return data_classify

    def dm_transform(self, data, freq, dt=8.192e-5, dm_max=10, 
                     dm_min=-10, ndm=64, freq_ref=None):
        """ Transform freq/time data to dm/time data.                                                    
        """

        if len(freq)==2:
            NFREQ = data.shape[0]
            freq = np.linspace(freq[0], freq[1], NFREQ)

        if len(data.shape)==2:
            ntab = 1

        nfreq = data.shape[1]
        ntime = data.shape[-1]
        ntab = data.shape[0]

        dms = np.linspace(dm_min, dm_max, ndm)

        data_full = np.zeros([ntab, ndm, ntime])
        times = np.linspace(-0.5*ntime*dt, 0.5*ntime*dt, ntime)

        for ii, dm in enumerate(dms):
            if len(data.shape)==2:
                data_full[:, ii] = np.mean(self.dedisperse(data.copy(), dm, freq=(freq[0], freq[-1]),
                                        freq_ref=freq_ref), axis=1)
            elif len(data.shape)==3:
                data_full[:, ii] = np.mean(self.dedisperse_tabs(data.copy(), dm, freq=(freq[0], freq[-1]),
                                        freq_ref=freq_ref), axis=1)
            else:
                raise Exception('Expected data shape: (ntab,nfreq,ntime) or (nfreq,ntime)')

        return data_full, dms, times    

    def proc_all(self, data, dm, nfreq_plot=32, ntime_plot=64, 
                 invert_spectrum=False, downsample=1, dmtransform=True, 
                 freq=(1550.,1250.)):
        t0 = time.time()
        data = self.preprocess(data, invert_spectrum=invert_spectrum, threshold=np.inf)
        #print('t preproc: %f' % (time.time()-t0))
        data = self.dedisperse_tabs(data, dm)
        #print('t dedisp_tabs: %f' % (time.time()-t0))
        data_classify_freqtime = self.postprocess(data, nfreq_plot=nfreq_plot, 
                                        ntime_plot=ntime_plot, downsample=downsample)
        #print('t postproc: %f' % (time.time()-t0))

        if dmtransform:
            data_dmtime, dms, times = self.dm_transform(data_classify_freqtime, freq=freq)
            data_dmtime -= np.median(data_dmtime)
            data_dmtime /= np.std(data_dmtime)
            data_dmtime[np.isnan(data_dmtime)] = 0.
        else:
            data_dmtime = []
        #print('t dmtrans: %f' % (time.time()-t0))

        return data_classify_freqtime, data_dmtime





