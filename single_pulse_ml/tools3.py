import numpy as np
import glob
import scipy.signal
import optparse 
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib as mpl
    mpl.use('Agg', warn=False)

# should there maybe be a clustering class
# and a S/N calculation class?

class AnalyseTriggers:

    def __init__(self):
        pass 

class RealtimeProc:

    def __int__(self, dt=8.192e-5):
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

    def proc_all(self, data, dm, nfreq_plot=32, ntime_plot=64, invert_spectrum=False, downsample=1):
        data = self.preprocess(data, invert_spectrum=invert_spectrum)
        data = self.dedisperse_tabs(data, dm)
        data_classify = self.postprocess(data, nfreq_plot=nfreq_plot, ntime_plot=ntime_plot, downsample=downsample)
        
        return data_classify



def combine_all_beams(fdir, fnout=None):

    flist = glob.glob(fdir+'/CB*.cand')

    data_all = []

    for fn in flist:
        print(fn)
        try:
            CB = float(fn.split('CB')[-1][:2])
        except:
            continue

        data = np.genfromtxt(fn)

        if len(data)<1:
            continue

        beamno = np.ones([len(data)])*CB
        data_full = np.concatenate([data, beamno[:, None]], axis=-1)
        data_all.append(data_full)

    data_all = np.concatenate(data_all)
    if type(fnout) is str:
        np.savetxt(fnout, data_all)

    return data_all

def get_multibeam_triggers(times, beamno, t_window=0.5):
    CB_list = set(beamno)
    nbins = int((times.max()-times.min())/t_window)
    ntrig_perbeam = np.zeros([nbins])

    for CB in CB_list:
        if CB==13:
            continue
        vals, time_bins = np.histogram(times[beamno==CB], 
                                       range=(times.min()-1, times.max()+1), 
                                       bins=nbins)
        vals[vals!=0] = 1.0
        ntrig_perbeam += vals

    return ntrig_perbeam

def dedisperse(data, dm, dt=8.192e-5, freq=(1550, 1250), freq_ref=None):
    data = data.copy()
    
    nfreq, ntime = data.shape[0], data.shape[1]

    freqs = np.linspace(freq[0], freq[-1], nfreq)

    if freq_ref is None:
        freq_ref = freqs.max()

    tdelay = 4.148e3*dm*(freqs**-2 - freq_ref**-2)
    ntime = len(data[0])

    maxind_arr = []

    for ii, f in enumerate(freqs):
        data[ii] = np.roll(data[ii], -np.int(tdelay[ii]/dt))

    return data

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

def cleandata(data, threshold=3.0):
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

def group_dm_time_beam(fdir, fnout=None, trigname='cand'):
    """ Go through all compound beams (CB) in 
    directory fdir, group in time/DM, then 
    group in time/CB. 
    """

    flist = glob.glob(fdir+'/CB*%s' % trigname)

    times_full, beamno_full, dm_full = [], [], []
    for fn in flist:
        print(fn)
        try:
            CB = float(fn.split('CB')[-1][:2])
        except:
            continue

        try:
            sig_cut, dm_cut, tt_cut, ds_cut, ind_full = \
                         get_triggers(fn, sig_thresh=8.0, 
                         dm_min=10.0, dm_max=np.inf, 
                         t_window=0.5, 
                         max_rows=None)
        except IndexError:
            print(("Skipping CB%d" % CB))
            continue

        beamno = np.ones([len(dm_cut)])*CB

        times_full.append(tt_cut)
        beamno_full.append(beamno)
        dm_full.append(dm_cut)

    times_full = np.concatenate(times_full)
    beamno_full = np.concatenate(beamno_full)
    dm_full = np.concatenate(dm_full)

    ntrig_pb = get_multibeam_triggers(times_full, beamno_full, t_window=1.0)

    return times_full, beamno_full, dm_full, ntrig_pb

def dm_range(dm_max, dm_min=5., frac=0.2):
    """ Generate list of DM-windows in which 
    to search for single pulse groups. 

    Parameters
    ----------
    dm_max : float 
        max DM 
    dm_min : float  
        min DM 
    frac : float 
        fractional size of each window 

    Returns
    -------
    dm_list : list 
        list of tuples containing (min, max) of each 
        DM window
    """

    dm_list =[]
    prefac = (1-frac)/(1+frac)

    while dm_max>dm_min:
        if dm_max < 100.:
            prefac = (1-2*frac)/(1+2*frac)
        if dm_max < 50.:
            prefac = 0.0 

        dm_list.append((int(prefac*dm_max), int(dm_max)))
        dm_max = int(prefac*dm_max)

    return dm_list

def read_singlepulse(fn, max_rows=None, beam=None):
    """ Read in text file containing single-pulse 
    candidates. Allowed formats are:
    .singlepulse = PRESTO output
    .txt = injection pipeline output
    .trigger = AMBER output 
    .cand = heimdall output 

    max_rows sets the maximum number of 
    rows to read from textfile 
    beam is the beam number to pick in case of .trigger files
    """

    if fn.split('.')[-1] in ('singlepulse', 'txt'):
        A = np.genfromtxt(fn, max_rows=max_rows)

        if len(A.shape)==1:
            A = A[None]

        dm, sig, tt, downsample = A[:,0], A[:,1], A[:,2], A[:,4]
    elif fn.split('.')[-1]=='trigger':
        A = np.genfromtxt(fn, max_rows=max_rows)

        if len(A.shape)==1:
            A = A[None]

        # Check if amber has compacted, in which case 
        # there are two extra rows
        if len(A[0]) > 7:
            if len(A[0])==8:
                # beam batch sample integration_step compacted_integration_steps time DM compacted_DMs SNR
                beamno, dm, sig, tt, downsample = A[:, 0], A[:,-3], A[:,-1], A[:, -4], A[:, 3]
            elif len(A[0])==10:
                beamno, dm, sig, tt, downsample = A[:, 0], A[:,-3], A[:,-1], A[:, -5], A[:, 3]
            else:
                print("Error: DO NOT RECOGNIZE COLUMNS OF .trigger FILE")
                return 
        else:
            # beam batch sample integration_step time DM SNR
            beamno, dm, sig, tt, downsample = A[:, 0], A[:,-2], A[:,-1], A[:, -3], A[:, 3]
        
        if beam is not None:
            # pick only the specified beam
            dm = dm[beamno.astype(int) == beam]
            sig = sig[beamno.astype(int) == beam]
            tt = tt[beamno.astype(int) == beam]
            downsample = downsample[beamno.astype(int) == beam]

    elif fn.split('.')[-1]=='cand':
        A = np.genfromtxt(fn, max_rows=max_rows)

        if len(A.shape)==1:
            A = A[None]

        # SNR sample_no time log_2_width DM_trial DM Members first_samp last_samp
        dm, sig, tt, log_2_downsample = A[:,5], A[:,0], A[:, 2], A[:, 3]
        downsample = 2**log_2_downsample
        try:
            beamno = A[:, 9]
            return dm, sig, tt, downsample, beamno
        except:
            pass
    else:
        print("Didn't recognize singlepulse file")
        return 

    if len(A)==0:
        return 0, 0, 0, 0

    return dm, sig, tt, downsample

def get_triggers(fn, sig_thresh=5.0, dm_min=0, dm_max=np.inf, 
                 t_window=0.5, max_rows=None, t_max=np.inf,
                 sig_max=np.inf, dt=2*40.96, delta_nu_MHz=300./1536, 
                 nu_GHz=1.4, fnout=False, tab=None, dm_width_filter=False):
    """ Get brightest trigger in each 10s chunk.

    Parameters
    ----------
    fn : str 
        filename with triggers (.npy, .singlepulse, .trigger)
    sig_thresh : float
        min S/N to include
    dm_min : 
        minimum dispersion measure to allow 
    dm_max : 
        maximum dispersion measure to allow 
    t_window : float 
        Size of each time window in seconds
    max_rows : 
        Only read this many rows from raw trigger file 
    fnout : str 
        name of text file to save clustered triggers to 
    tab : int
        which TAB to process (0 for IAB)

    Returns
    -------
    sig_cut : ndarray
        S/N array of brightest trigger in each DM/T window 
    dm_cut : ndarray
        DMs of brightest trigger in each DM/T window 
    tt_cut : ndarray
        Arrival times of brightest trigger in each DM/T window 
    ds_cut : ndarray 
        downsample factor array of brightest trigger in each DM/T window 
    """
    if tab is not None:
        beam_amber = tab
    else:
        beam_amber = None

    if type(fn)==str:
        dm, sig, tt, downsample = read_singlepulse(fn, max_rows=max_rows, beam=beam_amber)[:4]
    elif type(fn)==np.ndarray:
        dm, sig, tt, downsample = fn[:,0], fn[:,1], fn[:,2], fn[:,3]
    else:
        print("Wrong input type. Expected string or nparray")
        return [],[],[],[],[]

    ntrig_orig = len(dm)

    bad_sig_ind = np.where((sig < sig_thresh) | (sig > sig_max))[0]
    sig = np.delete(sig, bad_sig_ind)
    tt = np.delete(tt, bad_sig_ind)
    dm = np.delete(dm, bad_sig_ind)
    downsample = np.delete(downsample, bad_sig_ind)
    sig_cut, dm_cut, tt_cut, ds_cut = [],[],[],[]

    if len(tt)==0:
        print("Returning None: time array is empty")
        return 

    tduration = tt.max() - tt.min()
    ntime = int(tduration / t_window)

    # Make dm windows between 90% of the lowest trigger and 
    # 10% of the largest trigger
    if dm_min==0:
        dm_min = 0.9*dm.min()
    if dm_max > 1.1*dm.max():
        dm_max = 1.1*dm.max()

    # Can either do the DM selection here, or after the loop
#    dm_list = dm_range(dm_max, dm_min=dm_min)
    dm_list = dm_range(1.1*dm.max(), dm_min=0.9*dm.min())

    print(("\nGrouping in window of %.2f sec" % np.round(t_window,2)))
    print(("DMs:", dm_list))

    tt_start = tt.min() - .5*t_window
    ind_full = []

    # might wanna make this a search in (dm,t,width) cubes
    for dms in dm_list:
        for ii in range(ntime + 2):
            try:    
                # step through windows of t_window seconds, starting from tt.min()
                # and find max S/N trigger in each DM/time box
                t0, tm = t_window*ii + tt_start, t_window*(ii+1) + tt_start
                ind = np.where((dm<dms[1]) & (dm>dms[0]) & (tt<tm) & (tt>t0))[0] 
                ind_maxsnr = ind[np.argmax(sig[ind])]
                sig_cut.append(sig[ind_maxsnr])
                dm_cut.append(dm[ind_maxsnr])
                tt_cut.append(tt[ind_maxsnr])
                ds_cut.append(downsample[ind_maxsnr])
                ind_full.append(ind_maxsnr)
            except:
                continue

    ind_full = np.array(ind_full)
    dm_cut = np.array(dm_cut)
    # now remove the low DM candidates
    tt_cut = np.array(tt_cut).astype(np.float)
    ind = np.where((dm_cut >= dm_min) & (dm_cut <= dm_max) & (tt_cut < t_max))[0]

    dm_cut = dm_cut[ind]
    ind_full = ind_full[ind]
    sig_cut = np.array(sig_cut)[ind]
    tt_cut = tt_cut[ind]
    ds_cut = np.array(ds_cut)[ind]

    ntrig_group = len(dm_cut)

    print(("Grouped down to %d triggers from %d\n" % (ntrig_group, ntrig_orig)))

    rm_ii = []

    if dm_width_filter:
        for ii in range(len(ds_cut)):        
            tdm = 8.3 * delta_nu_MHz / nu_GHz**3 * dm_cut[ii] # microseconds#

            if ds_cut[ii]*dt < (0.5*(dt**2 + tdm**2)**0.5):
                rm_ii.append(ii)

    dm_cut = np.delete(dm_cut, rm_ii)
    tt_cut = np.delete(tt_cut, rm_ii)
    sig_cut = np.delete(sig_cut, rm_ii)
    ds_cut = np.delete(ds_cut, rm_ii)
    ind_full = np.delete(ind_full, rm_ii)

    if fnout != False:
        clustered_arr = np.concatenate([sig_cut, dm_cut, tt_cut, ds_cut, ind_full])
        clustered_arr = clustered_arr.reshape(5, -1)
        np.savetxt(fnout, clustered_arr) 

    return sig_cut, dm_cut, tt_cut, ds_cut, ind_full

def add_tab_col(fdir, fnout='out'):
    """ Take list of .trigger files for 
    all TABs, concanetate into single .trigger
    file with correct TAB column.
    """
    fl = glob.glob(fdir)
    fl.sort()
    trigg_arr_full = []
    
    ext = fl[0].split('.')[-1]
    assert ext=='trigger', 'expected an amber output file .trigger'

    for ff in fl:
        tab = int(ff.split('CB')[-1].split('_')[1][:2])
        trigg_arr = np.loadtxt(ff)
        trigg_arr[:, 0] = tab
        trigg_arr_full.append(trigg_arr)
    
    trigg_arr_full = np.concatenate(trigg_arr_full)
    np.savetxt(fnout+'.'+ext, trigg_arr_full)

def plot_tab_summary(fn, ntab=12, suptitle=''):
    fig, axs = plt.subplots(6, 4, sharex=True, figsize=(12,10))
    fig.subplots_adjust(hspace=0)

    ntot = 0
    for tab in range(ntab):
        try:
            sig_cut, dm_cut, tt_cut, ds_cut, ind_full = get_triggers(fn, tab=tab)
        except(TypeError):
            print(("No triggers from Tab %d" % tab))

        subind1 = (1+tab+4*(tab//4))
        subind2 = (1+tab+4*(tab//4)+4)

        if subind1 in [1, 9, 17]:
            yl1 = 'log frac'
            yl2 = 'Time'
        else:
            yl1 = ''
            yl2 = ''

        ntot += len(sig_cut)

        plt.subplot(6,4,subind1)
        plt.hist(np.log10(dm_cut), bins=30, log=True, color='C1', alpha=0.5)
        plt.legend(['TAB %d' % tab], loc=2)
        plt.yticks([])
        plt.ylabel(yl1, fontsize=14)
        plt.xlim(-1, 3.5)

        plt.subplot(6,4,subind2)
        plt.scatter(np.log10(dm_cut), tt_cut, np.log10(sig_cut), color='k')
        plt.yticks([])
        plt.ylabel(yl2, fontsize=14)
        plt.xlim(-1, 3.5)

        if tab > 7:
            plt.xlabel('log10(DM)', fontsize=14)

    suptitle_ = suptitle + '\nTotal triggers: %d' % ntot 
    plt.suptitle(suptitle_, fontsize=20)
    plt.show()

    fig, axs = plt.subplots(6, 4, sharex=True, figsize=(12,10))
    fig.subplots_adjust(hspace=0)
    ntot = 0
    for tab in range(ntab):
        try:
            sig_cut, dm_cut, tt_cut, ds_cut, ind_full = get_triggers(fn, tab=tab)
        except(TypeError):
            print(("No triggers from Tab %d" % tab))

        subind1 = 1 + tab #(1+tab+4*(tab//4))
        subind2 = 1 + tab + 12#(1+tab+4*(tab//4)+4)

        ntot += len(sig_cut)

        plt.subplot(6,4,subind1)
        plt.hist(np.log2(ds_cut), bins=30, log=True, color='C0', alpha=0.5)
        plt.legend(['TAB %d' % tab], loc=2)
        plt.yticks([])
        plt.ylabel(yl1, fontsize=14)

        if subind1 > 8:
            plt.xlabel('log2(Width)', fontsize=14)

        plt.subplot(6,4,subind2)
        plt.hist(np.log10(sig_cut), bins=30, log=True, color='C1', alpha=0.5)
        plt.yticks([])
        plt.ylabel(yl2, fontsize=14)
        plt.legend(['TAB %d' % tab], loc=2)

        if subind2 > 20:
            plt.xlabel('log10(S/N)', fontsize=14)

    suptitle_ = suptitle + '\nTotal triggers: %d' % ntot 
    plt.suptitle(suptitle_, fontsize=20)
    plt.show()

class SNR_Tools:

    def __init__(self):
        pass

    def sigma_from_mad(self, data):
        """ Get gaussian std from median 
        aboslute deviation (MAD)
        """
        assert len(data.shape)==1, 'data should be one dimensional'

        med = np.median(data)
        mad = np.median(np.absolute(data - med))

        return 1.4826*mad, med

    def calc_snr_presto(self, data):
        """ Calculate S/N of 1D input array (data)
        after excluding 0.05 at tails
        """
        std_chunk = scipy.signal.detrend(data, type='linear')
        std_chunk.sort()
        ntime_r = len(std_chunk)
        stds = 1.148*np.sqrt((std_chunk[ntime_r//40:-ntime_r//40]**2.0).sum() /
                              (0.95*ntime_r))
        snr_ = std_chunk[-1] / stds 

        return snr_

    def calc_snr_amber(self, data, thresh=3.):
        sig = np.std(data)
        dmax = (data.copy()).max()
        dmed = np.median(data)
        N = len(data)

        # remove outliers 4 times until there 
        # are no events above threshold*sigma
        for ii in range(4):
            ind = np.where(np.abs(data-dmed)<thresh*sig)[0]
            sig = np.std(data[ind])
            dmed = np.median(data[ind])
            data = data[ind]
            N = len(data)

        snr_ = (dmax - dmed)/(1.048*sig)

        return snr_

    def calc_snr_mad(self, data):
        sig, med = self.sigma_from_mad(data)

        return (data.max() - med) / sig

    def calc_snr_matchedfilter(self, data, widths=None, true_filter=None):
        """ Calculate the S/N of pulse profile after 
        trying 9 rebinnings.

        Parameters
        ----------
        arr   : np.array
            (ntime,) vector of pulse profile 

        Returns
        -------
        snr : np.float 
            S/N of pulse
        """
        assert len(data.shape)==1
        
        ntime = len(data)
        snr_max = 0
        width_max = 0
#        data = scipy.signal.detrend(data, type='linear')

        if widths is None:
            widths = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500]

        for ii in widths:
            if true_filter is None:
                mf = np.ones([ii])
            else:
                mf = true_filter
            data_mf = scipy.correlate(data, mf)
            snr_ = self.calc_snr_amber(data_mf)

            if snr_ > snr_max:
                snr_max = snr_
                width_max = ii

        return snr_max, width_max
        

    def calc_snr_widths(self, data, widths=None):
        """ Calculate the S/N of pulse profile after 
        trying 9 rebinnings.

        Parameters
        ----------
        arr   : np.array
            (ntime,) vector of pulse profile 

        Returns
        -------
        snr : np.float 
            S/N of pulse
        """
        assert len(data.shape)==1
        
        ntime = len(data)
        snr_max = 0
        data -= np.median(data)

        if widths is None:
            widths = [1, 2, 4, 8, 16, 32, 64, 128]

    #    for ii in range(1, 10):
        for ii in widths:
            for jj in range(ii):
                # skip if boxcar width is greater than 1/4th ntime
                if ii > ntime//8:
                    continue
                
                arr_copy = data.copy()
                arr_copy = np.roll(arr_copy, jj)
                arr_ = arr_copy[:ntime//ii*ii].reshape(-1, ii).mean(-1)

                snr_ = self.calc_snr(arr_)

                if snr_ > snr_max:
                    snr_max = snr_
                    width_max = ii

        return snr_max, width_max

    def compare_snr(self, fn_1, fn_2, dm_min=0, dm_max=np.inf, save_data=False,
                    sig_thresh=5.0, t_window=0.5, max_rows=None,
                    t_max=np.inf, tab=None, freq_ref_1=1400., freq_ref_2=1400.):
        """ Read in two files with single-pulse candidates
        and compare triggers.

        Parameters:
        ----------
        fn_1 : str 
            name of input triggers text file
            (must be .trigger, .singlepulse, or .txt)
        fn_2 : str
            name of input triggers text file for comparison 
        dm_min : float
            do not process triggers below this DM 
        dm_max : float 
            do not process triggers above this DM 
        save_data : bool 
            if True save to np.array
        sig_thresh : float 
            do not process triggers below this S/N 
        t_window : float 
            time window within which triggers in 
            fn_1 and fn_2 will be considered the same 

        Return:
        -------
        Function returns four parameter arrays for 
        each fn_1 and fn_2, which should be ordered so 
        that they can be compared directly:

        grouped_params1, grouped_params2, matched_params
        """
        snr_1, dm_1, t_1, w_1, ind_full_1 = get_triggers(fn_1, sig_thresh=sig_thresh, 
                                    dm_min=dm_min, dm_max=dm_max, t_window=t_window, 
                                                         max_rows=max_rows, t_max=t_max, tab=tab)

        snr_2, dm_2, t_2, w_2, ind_full_2 = get_triggers(fn_2, sig_thresh=sig_thresh, 
                                    dm_min=dm_min, dm_max=dm_max, t_window=t_window, 
                                                         max_rows=max_rows, t_max=t_max, tab=tab)

        # adjust arrival times to have same ref freq after dedispersion
        t_1 += 4148*dm_1*(freq_ref_2**-2 - freq_ref_1**-2)

        snr_2_reorder = []
        dm_2_reorder = []
        t_2_reorder = []
        w_2_reorder = []

        ntrig_1 = len(snr_1)
        ntrig_2 = len(snr_2)    

        par_1 = np.concatenate([snr_1, dm_1, t_1, w_1, ind_full_1]).reshape(5, -1)
        par_2 = np.concatenate([snr_2, dm_2, t_2, w_2, ind_full_2]).reshape(5, -1)

        # Make arrays for the matching parameters
        par_match_arr = []
        ind_missed = []
        ind_matched = []

        for ii in range(len(snr_1)):

            tdiff = np.abs(t_1[ii] - t_2)
            ind = np.where(tdiff == tdiff.min())[0]

            if t_1[ii] > t_max:
                continue

            # make sure you are getting correct trigger in dm/time space
            if len(ind) > 1:
                ind = ind[np.argmin(np.abs(dm_1[ii]-dm_2[ind]))]
            else:
                ind = ind[0]

            # check for triggers that are within 1.0 seconds and 20% in dm
            if (tdiff[ind]<1.0) and (np.abs(dm_1[ii]-dm_2[ind])/dm_1[ii])<0.2:
                pparams = (tdiff[ind], t_1[ii], t_2[ind], dm_1[ii], dm_2[ind], snr_1[ii], snr_2[ind], w_1[ii], w_2[ind])
                print(("%1.4f  %5.1f  %5.1f  %5.1f  %5.1f %5.1f  %5.1f %5.1f  %5.1f" % pparams))

                params_match = np.array([snr_1[ii], snr_2[ind], 
                                         dm_1[ii], dm_2[ind],
                                         t_1[ii], t_2[ind],
                                         w_1[ii], w_2[ind]])

                par_match_arr.append(params_match)
                ind_matched.append(ii)

            else:
                # Keep track of missed triggers
                ind_missed.append(ii)

        if len(par_match_arr)==0:
            print("No matches found")
            return 

        # concatenate list and reshape to (nparam, nmatch, 2 files)
        par_match_arr = np.concatenate(par_match_arr).reshape(-1, 4, 2)
        par_match_arr = par_match_arr.transpose((1, 0, 2))

        if save_data is True:
            nsnr = min(len(snr_1), len(snr_2))
            snr_1 = snr_1[:nsnr]
            snr_2 = snr_2_reorder[:nsnr]

            np.save(fn_1+'_params_grouped', par_1)
            np.save(fn_2+'_params_grouped', par_2)
            np.save('params_matched', par_match_1)

        return par_1, par_2, par_match_arr, ind_missed, ind_matched  

    def plot_comparison(self, par_1, par_2, par_match_arr, ind_missed, figname='./test.pdf'):
        fig = plt.figure(figsize=(14,14))

        frac_recovered = len(ind_missed)

        snr_1, snr_2 = par_1[0], par_2[0]
        dm_1, dm_2 = par_1[1], par_2[1]
        width_1, width_2 = par_1[3], par_2[3]

        snr_1_match = par_match_arr[0,:,0]
        snr_2_match = par_match_arr[0,:,1]

        dm_1_match = par_match_arr[1,:,0]
        dm_2_match = par_match_arr[1,:,1]

        width_1_match = par_match_arr[3,:,0]
        width_2_match = par_match_arr[3,:,1]

        fig.add_subplot(311)
        plt.plot(snr_1_match, snr_2_match, '.')
        plt.plot(snr_1, snr_1, color='k')
        plt.plot(snr_1[ind_missed], np.zeros([len(ind_missed)]), 'o', color='orange')
        plt.xlabel('Injected S/N', fontsize=13)
        plt.ylabel('Detected S/N', fontsize=13)        
        plt.legend(['Detected events','Expected S/N','Missed events'], fontsize=13)

        fig.add_subplot(312)
        plt.plot(dm_1_match, snr_1_match/snr_2_match, '.')
        plt.plot(dm_1[ind_missed], np.zeros([len(ind_missed)]), 'o', color='orange')
        plt.xlabel('DM', fontsize=13)
        plt.ylabel('Expected S/N : Detected S/N', fontsize=13)        
        plt.legend(['Detected events','Missed events'], fontsize=13)

        fig.add_subplot(337)
        plt.hist(width_1, bins=50, alpha=0.3, normed=True)
        plt.hist(width_2, bins=50, alpha=0.3, normed=True)
        plt.hist(width_1[ind_missed], bins=50, alpha=0.3, normed=True)
        plt.xlabel('Width [samples]', fontsize=13)

        fig.add_subplot(338)
        plt.plot(width_1_match, snr_1_match,'.')
        plt.plot(width_1_match, snr_2_match,'.')
        plt.plot(width_1, snr_1, '.')
        plt.xlabel('Width [samples]', fontsize=13)
        plt.ylabel('S/N injected', fontsize=13)

        fig.add_subplot(339)
        plt.plot(width_1_match, dm_1_match,'.')
        plt.plot(width_1_match, dm_2_match,'.')
        plt.plot(width_1, dm_1,'.')
        plt.xlabel('Width [samples]', fontsize=13)
        plt.ylabel('DM', fontsize=13)

        plt.tight_layout()
        plt.show()
        plt.savefig(figname)


if __name__=='__main__':

    import sys

    SNRTools = SNR_Tools()

    parser = optparse.OptionParser(prog="tools.py", \
                        version="", \
                        usage="%prog fn1 fn2 [OPTIONS]", \
                        description="Compare to single-pulse trigger files")

    parser.add_option('--sig_thresh', dest='sig_thresh', type='float', \
                        help="Only process events above >sig_thresh S/N" \
                                "(Default: 5.0)", default=5.0)

    parser.add_option('--save_data', dest='save_data', type='str',
                        help="save each trigger's data. 0=don't save. \
                        hdf5 = save to hdf5. npy=save to npy. concat to \
                        save all triggers into one file",
                        default='hdf5')

    parser.add_option('--mk_plot', dest='mk_plot', action='store_true', \
                        help="make plot if True (default False)", default=False)

    parser.add_option('--dm_min', dest='dm_min', type='float',
                        help="", 
                        default=0.0)

    parser.add_option('--dm_max', dest='dm_max', type='float',
                        help="", 
                        default=np.inf)
    parser.add_option('--t_max', dest='t_max', type='float',
                        help="Only process first t_max seconds", 
                        default=np.inf)
    parser.add_option('--t_window', dest='t_window', type='float',
                        help="", 
                        default=0.1)
    parser.add_option('--outdir', dest='outdir', type='str',
                        help="directory to write data to", 
                        default='./data/')
    parser.add_option('--title', dest='title', type='str',
                        help="directory to write data to", 
                        default='file1 vs. file2')
    parser.add_option('--figname', dest='figname', type='str',
                        help="directory to write data to", 
                        default='comparison.pdf')
    parser.add_option('--algo1', dest='algo1', type='str',
                        help="name of first algo", 
                        default='algorithm1')
    parser.add_option('--algo2', dest='algo2', type='str',
                        help="name of second algo", 
                        default='algorithm2')
    parser.add_option('--truthfile', dest='truthfile', type='str',
                        help="truth file", 
                        default=None)
    parser.add_option('--tab', dest='tab', type=int, \
                        help="TAB to process (0 for IAB) (default: 0)", default=0)
    parser.add_option('--plot_both', dest='plot_both', action='store_true', \
                        help="make plot with both fn1 vs. fn2 and fn2 vs. fn1", default=False)
    parser.add_option('--freq_ref_1', dest='freq_ref_1', type=float, \
                        help="Reference frequency of fn1", default=1400.)
    parser.add_option('--freq_ref_2', dest='freq_ref_2', type=float, \
                        help="Reference frequency of fn2", default=1400.)


    options, args = parser.parse_args()
    fn_1 = args[0]
    fn_2 = args[1]

    try:
        par_1a, par_2a, par_match_arra, ind_misseda, ind_matcheda = SNRTools.compare_snr(fn_1, fn_2, 
                                        dm_min=options.dm_min, 
                                        dm_max=options.dm_max, save_data=False,
                                        sig_thresh=options.sig_thresh, 
                                        t_window=options.t_window, 
                                        max_rows=None, t_max=options.t_max,
                                                                                         tab=options.tab, freq_ref_1=options.freq_ref_1, freq_ref_2=options.freq_ref_2)
        if options.plot_both is True:
            par_1b, par_2b, par_match_arrb, ind_missedb, ind_matchedb = SNRTools.compare_snr(fn_2, fn_1, 
                                        dm_min=options.dm_min, 
                                        dm_max=options.dm_max, save_data=False,
                                        sig_thresh=options.sig_thresh, 
                                        t_window=options.t_window, 
                                        max_rows=None, t_max=options.t_max, 
                                                                                             tab=options.tab, freq_ref_1=options.freq_ref_2, freq_ref_2=options.freq_ref_1)                                       

    except TypeError:
        print("No matches, exiting")
        exit()
        
    print(('\nFound %d common trigger(s)' % par_match_arra.shape[1]))

    snr_1 = par_match_arra[0, :, 0]
    snr_2 = par_match_arra[0, :, 1]

    print(('\nFile 1 has %f times higher S/N than file 2\n' % np.mean(snr_1/snr_2)))

    mk_plot = True

    if options.mk_plot is True:

        import plotter 
        plotter.plot_comparison(par_1a, par_2a, par_match_arra, ind_misseda, 
                                figname=options.figname,
                                algo1=options.algo1, algo2=options.algo2)

        if options.plot_both is True:
            plotter.plot_comparison(par_1b, par_2b, par_match_arrb, ind_missedb, 
                                figname=options.figname, 
                                algo1=options.algo2, algo2=options.algo1)

        if options.truthfile is not None:
            par_1, par_1_truth, par_match_1, ind_misseda, ind_matched1 = \
                                        SNRTools.compare_snr(fn_1, options.truthfile, 
                                        dm_min=options.dm_min, 
                                        dm_max=options.dm_max, save_data=False,
                                        sig_thresh=options.sig_thresh, 
                                        t_window=options.t_window, 
                                                             max_rows=None, t_max=options.t_max, 
                                                             tab=options.tab)

            par_2, par_2_truth, par_match_2, ind_missedb, ind_matched2 = \
                                        SNRTools.compare_snr(fn_2, options.truthfile, 
                                        dm_min=options.dm_min, 
                                        dm_max=options.dm_max, save_data=False,
                                        sig_thresh=options.sig_thresh, 
                                        t_window=options.t_window, 
                                                             max_rows=None, t_max=options.t_max, 
                                                             tab=options.tab)                                       
 
            plotter.plot_against_truth(par_match_1, par_match_2)

            print("Comparing both against the truth file")
            
            
#        SNRTools.plot_comparison(par_1, par_2, par_match_arr, ind_missed, figname=figname)













