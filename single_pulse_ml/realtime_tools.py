import numpy as np

class DadaHeader:

    def __init__(self, header):
        self.dt = np.float(header['TSAMP'])
        self.nchan = np.int(header['NCHAN'])
        self.ntime_batch = np.int(header['PADDED_SIZE'])
        self.freq_mid = np.float(header['FREQ']) # mid frequency?
        self.freq_low = np.float(header['MIN_FREQUENCY'])
        self.dnu = np.float(header['CHANNEL_BANDWIDTH'])
        self.freq_high = self.freq_low + self.nchan*self.dnu

class RealtimeProc:

    def __init__(self, dt=8.192e-5):
        self.dt = dt

    def cleandata(self, data, threshold=3.0):
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

        assert len(data.shape)==2, "Expected (nfreq, ntime) array"

        dtmean = np.mean(data, axis=-1)
        dfmean = np.mean(data, axis=0)
        stdevf = np.std(dfmean)
        medf = np.median(dfmean)
        maskf = np.where(np.abs(dfmean - medf) > threshold*stdevf)[0]        

        # replace with mean spectrum
        data[:, maskf] = dtmean[:, None]*np.ones(len(maskf))[None]

        return data

    def dedisperse(self, data, dm, freq=(1550, 1250), freq_ref=None):
        print(data.shape)
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

    def preprocess(self, data, invert_spectrum=False):
        if len(data.shape)==2:
            data = data[None]

        if invert_spectrum:
            data = data[:, ::-1]

        ntab = data.shape[0]

        for tab in range(ntab):
            data[tab] = self.cleandata(data[tab], threshold=3.0)

        if data.shape[0]==1:
            data = data[0]

        return data 

    def dedisperse_tabs(self, data, dm, dt=8.192e-5, freq=(1550, 1250), freq_ref=None):

        if len(data.shape)==2:
            data = data[None]

        ntab = data.shape[0]

        for tab in range(ntab):
            data[tab] = self.dedisperse(data[tab], dm, freq=freq, freq_ref=freq_ref)

        if data.shape[0]==1:
            data = data[0]

        return data

    def postprocess(self, data, nfreq_plot=32, ntime_plot=64, downsample=1):
        if len(data.shape)==2:
            data = data[None]

        nfreq = data.shape[1]
        ntime = data.shape[-1]
        ntab = data.shape[0]
        data_classify = np.empty([ntab, nfreq_plot, ntime_plot])

        for tab in range(ntab):
            data_tab = data[tab]
            data_tab = data_tab.reshape(nfreq_plot, nfreq//nfreq_plot, -1).mean(1)
            data_tab = data_tab[:, :ntime//downsample*downsample]
            data_tab = data_tab.reshape(-1, ntime//downsample, downsample).mean(-1)

            maxind = np.argmax(data_tab.mean(0))

            if (ntime-maxind)<ntime_plot//2:
                maxind = ntime - ntime_plot//2
            if maxind<ntime_plot//2:
                maxind = ntime_plot//2

            data_tab -= np.median(data_tab)
            data_tab /= np.std(data_tab)
            data_tab[data_tab!=data_tab] = 0.

            data_classify[tab] = data_tab[:, maxind-ntime_plot//2:maxind+ntime_plot//2]

        return data_classify

def dm_transform(data, freq, dt=8.192e-5, dm_max=10, dm_min=-10, ndm=50, freq_ref=None):
    """ Transform freq/time data to dm/time data.                                                    
    """

    if len(freq)<3:
        NFREQ = data.shape[0]
        freq = np.linspace(freq[0], freq[1], NFREQ)

    dms = np.linspace(dm_min, dm_max, ndm)
    ntime = data.shape[-1]

    data_full = np.zeros([ndm, ntime])
    times = np.linspace(-0.5*ntime*dt, 0.5*ntime*dt, ntime)

    for ii, dm in enumerate(dms):
        data_full[ii] = np.mean(dedisperse(data, dm, freq=(freq[0], freq[-1]),
                               freq_ref=freq_ref), axis=0)

    return data_full, dms, times    

    def proc_all(self, data, dm, nfreq_plot=32, ntime_plot=64, invert_spectrum=False, downsample=1):
        data = self.preprocess(data, invert_spectrum=invert_spectrum)
        data = self.dedisperse_tabs(data, dm)
        data_classify = self.postprocess(data, nfreq_plot=nfreq_plot, 
                                        ntime_plot=ntime_plot, downsample=downsample)
        data_dmt = self.dmt(data_classify)

        return data_classify, data_dmt
