import random

import numpy as np
import glob
from scipy import signal

try:
    import matplotlib.pyplot as plt
except:
    plt = None
    pass

import reader
import dataproc
import tools 

try:
    import plot_tools
except:
    plot_tools = None

# To do: 
# Put things into physical units. Scattering measure, actual widths, fluences, etc. 
# Need inputs of real telescopes. Currently it's vaguely like the Pathfinder.
# Need to not just simulate noise for the FRB triggers. 
# More single pixel widths. Unresolved bursts.
# Inverse fluence relationship right now! 

class Event(object):
    """ Class to generate a realistic fast radio burst and 
    add the event to data. 
    """
    def __init__(self, t_ref, f_ref, dm, fluence, width, 
                 spec_ind, disp_ind=2, scat_factor=0):
        self._t_ref = t_ref
        self._f_ref = f_ref
        self._dm = dm
        self._fluence = fluence 
        self._width = width
        self._spec_ind = spec_ind
        self._disp_ind = disp_ind
        self._scat_factor = min(1, scat_factor + 1e-18) # quick bug fix hack

    def disp_delay(self, f, _dm, _disp_ind=-2.):
        """ Calculate dispersion delay in seconds for 
        frequency,f, in MHz, _dm in pc cm**-3, and 
        a dispersion index, _disp_ind. 
        """
        return 4.148808e3 * _dm * (f**(-_disp_ind))

    def arrival_time(self, f):
        t = self.disp_delay(f, self._dm, self._disp_ind)
        t = t - self.disp_delay(self._f_ref, self._dm, self._disp_ind)
        return self._t_ref + t


    def calc_width(self, dm, freq_c, bw=400.0, NFREQ=1024,
                   ti=1, tsamp=1, tau=0):

        delta_freq = bw/NFREQ

        # taudm in milliseconds
        tdm = 8.3e-3 * dm * delta_freq / freq_c**3
        tI = np.sqrt(ti**2 + tsamp**2 + tdm**2 + tau**2)

        return tI

    def dm_smear(self, DM, freq_c, bw=400.0, NFREQ=1024,
                 ti=1, tsamp=0.0016, tau=0):  
        """ Calculate DM smearing SNR reduction
        """
        tau *= 1e3 # make ms
        ti *= 1e3 
        tsamp *= 1e3

        delta_freq = bw / NFREQ

        tI = np.sqrt(ti**2 + tsamp**2 + (8.3 * DM * delta_freq / freq_c**3)**2)

        return (np.sqrt(ti**2 + tau**2) / tI)**0.5

    def scintillation(self, freq):
        """ Include spectral scintillation across 
        the band. Approximate effect as a sinusoid, 
        with a random phase and a random decorrelation 
        bandwidth. 
        """
        # Make location of peaks / troughs random
        scint_phi = np.random.rand()
        # Make number of scintils between 0 and 10 (ish)
        nscint = np.random.uniform(0, 10)
        nscint = np.exp(np.random.uniform(np.log(1e-3), np.log(10)))
#        nscint = 10 #hack

        return np.cos(nscint*(freq - self._f_ref)/self._f_ref + scint_phi)**2

    def gaussian_profile(self, nt, width, t0=0.):
        """ Use a normalized Gaussian window for the pulse, 
        rather than a boxcar.
        """
        t = np.linspace(-nt//2, nt//2, nt)
        g = np.exp(-(t-t0)**2 / width**2)

        if not np.all(g > 0):
            g += 1e-18

        g /= g.max()

        return g

    def scat_profile(self, nt, f, tau=1.):
        """ Include exponential scattering profile. 
        """
        tau_nu = tau * (f / self._f_ref)**-4.
        t = np.linspace(0., nt//2, nt)

        prof = 1 / tau_nu * np.exp(-t / tau_nu)
        return prof / prof.max()

    def pulse_profile(self, nt, width, f, tau=100., t0=0.):
        """ Convolve the gaussian and scattering profiles 
        for final pulse shape at each frequency channel.
        """
        gaus_prof = self.gaussian_profile(nt, width, t0=t0)
        scat_prof = self.scat_profile(nt, f, tau) 
#        pulse_prof = np.convolve(gaus_prof, scat_prof, mode='full')[:nt]
        pulse_prof = signal.fftconvolve(gaus_prof, scat_prof)[:nt]
    
        return pulse_prof

    def add_to_data(self, delta_t, freq, data):
        """ Method to add already-dedispersed pulse 
        to background noise data. Includes frequency-dependent 
        width (smearing, scattering, etc.) and amplitude 
        (scintillation, spectral index). 
        """

        NFREQ = data.shape[0]
        NTIME = data.shape[1]
        tmid = NTIME//2

        scint_amp = self.scintillation(freq)

        for ii, f in enumerate(freq):
            width_ = 1e-3 * self.calc_width(self._dm, self._f_ref*1e-3, 
                                     bw=400.0, NFREQ=NFREQ,
                                     ti=self._width, tsamp=delta_t, tau=0)

#            width_ = self.dm_smear(self._dm, self._f_ref, 
#                                   delta_freq=400.0/1024, 
#                                   ti=self._width, tsamp=delta_t, tau=0)
            index_width = max(1, (np.round((width_/ delta_t))).astype(int))
            #index_width = max(1, (np.round((self._width/ delta_t))).astype(int))
            tpix = int(self.arrival_time(f) / delta_t)

            if abs(tpix) >= tmid:
                # ensure that edges of data are not crossed
                continue

            pp = self.pulse_profile(NTIME, index_width, f, 
                                    tau=self._scat_factor, t0=tpix)
            val = pp.copy()#[:len(pp)//NTIME * NTIME].reshape(NTIME, -1).mean(-1)
            val /= val.max()
            val *= self._fluence / self._width
            val = val * (f / self._f_ref) ** self._spec_ind 
            val = (0.25 + scint_amp[ii]) * val # hack
            data[ii] += val


    def dm_transform(self, delta_t, data, freq, maxdm=10.0, NDM=100):
        """ Transform freq/time data to dm/time data.
        """
    
        if len(freq)<3:
            NFREQ = data.shape[0]
            freq = np.linspace(freq[0], freq[1], NFREQ) 

        dm = np.linspace(-maxdm, maxdm, NDM)
        ndm = len(dm)
        ntime = data.shape[-1]

        data_full = np.zeros([ndm, ntime])

        for ii, dm in enumerate(dm):
            for jj, f in enumerate(freq):
                self._dm = dm
                tpix = int(self.arrival_time(f) / delta_t)
                data_rot = np.roll(data[jj], tpix, axis=-1)
                data_full[ii] += data_rot

        return data_full

class EventSimulator():
    """Generates simulated fast radio bursts.
    Events occurrences are drawn from a Poissonian distribution.
    """

    def __init__(self, dm=(0.,2000.), fluence=(0.03,0.3),
                 width=(2*0.0016, 1.), spec_ind=(-4.,4), 
                 disp_ind=2., scat_factor=(0, 0.5), freq=(800., 400.)):
        """
        Parameters
        ----------
        datasource : datasource.DataSource object
            Source of the data, specifying the data rate and band parameters.
        dm : float or pair of floats
            Burst dispersion measure or dispersion measure range (pc cm^-2).
        fluence : float or pair of floats
            Burst fluence (at band centre) or fluence range (s).
        width : float or pair of floats.
            Burst width or width range (s).
        spec_ind : float or pair of floats.
            Burst spectral index or spectral index range.
        disp_ind : float or pair of floats.
            Burst dispersion index or dispersion index range.
        freq : tuple 
            Min and max of frequency range in MHz. Assumes low freq 
            is first freq in array, not necessarily the lowest value. 

        """

        self.width = width
        self.freq_low = freq[0]
        self.freq_up = freq[1]

        if hasattr(dm, '__iter__') and len(dm) == 2:
            self._dm = tuple(dm)
        else:
            self._dm = (float(dm), float(dm))
        if hasattr(fluence, '__iter__') and len(fluence) == 2:
            self._fluence = tuple(fluence)
        else:
            self._fluence = (float(fluence), float(fluence))
        if hasattr(width, '__iter__') and len(width) == 2:
            self._width = tuple(width)
        else:
             self._width = (float(width), float(width))
        if hasattr(spec_ind, '__iter__') and len(spec_ind) == 2:
            self._spec_ind = tuple(spec_ind)
        else:
            self._spec_ind = (float(spec_ind), float(spec_ind))
        if hasattr(disp_ind, '__iter__') and len(disp_ind) == 2:
            self._disp_ind = tuple(disp_ind)
        else:
            self._disp_ind = (float(disp_ind), float(disp_ind))
        if hasattr(scat_factor, '__iter__') and len(scat_factor) == 2:
            self._scat_factor = tuple(scat_factor)
        else:
            self._scat_factor = (float(scat_factor), float(scat_factor))

        # self._freq = datasource.freq
        # self._delta_t = datasource.delta_t

        self._freq = np.linspace(self.freq_low, self.freq_up, 256) # tel parameter 

        self._simulated_events = []

        self._last_time_processed = 0.

    def draw_event_parameters(self):
        dm = uniform_range(*self._dm)
        fluence = uniform_range(*self._fluence)**(-2/3.)/0.5**(-2/3.)
        spec_ind = uniform_range(*self._spec_ind)
        disp_ind = uniform_range(*self._disp_ind)
        # turn this into a log uniform dist. Note not *that* many 
        # FRBs have been significantly scattered. Should maybe turn this 
        # knob down.
        scat_factor = np.exp(np.random.uniform(*self._scat_factor))
        # change width from uniform to lognormal
        width = np.random.lognormal(np.log(self._width[0]), self._width[1])
        width = max(min(width, 100*self._width[0]), 0.5*self._width[0])
        return dm, fluence, width, spec_ind, disp_ind, scat_factor

def uniform_range(min_, max_):
    return random.uniform(min_, max_)


def gen_simulated_frb(NFREQ=16, NTIME=250, sim=True, fluence=(0.03,0.3),
                spec_ind=(-4, 4), width=(2*0.0016, 1), dm=(-0.01, 0.01),
                scat_factor=(-3, -0.5), background_noise=None, delta_t=0.0016,
                plot_burst=False, freq=(800, 400), FREQ_REF=600., 
                ):
    """ Simulate fast radio bursts using the EventSimulator class.

    Parameters
    ----------
    NFREQ       : np.int 
        number of frequencies for simulated array
    NTIME       : np.int 
        number of times for simulated array
    sim         : bool 
        whether or not to simulate FRB or just create noise array
    spec_ind    : tuple 
        range of spectral index 
    width       : tuple 
        range of widths in seconds (atm assumed dt=0.0016)
    scat_factor : tuple 
        range of scattering measure (atm arbitrary units)
    background_noise : 
        if None, simulates white noise. Otherwise should be an array (NFREQ, NTIME)
    plot_burst : bool 
        generates a plot of the simulated burst

    Returns
    -------
    data : np.array 
        data array (NFREQ, NTIME)
    parameters : tuple 
        [dm, fluence, width, spec_ind, disp_ind, scat_factor]

    """
    plot_burst = False

    # Hard code incoherent Pathfinder data time resolution
    # Maybe instead this should take a telescope class, which 
    # has all of these things already.
    t_ref = 0. # hack

    if len(freq) < 3:
        freq=np.linspace(freq[0], freq[1], NFREQ)      

    if background_noise is None:
        # Generate background noise with unit variance
        data = np.random.normal(0, 1, NTIME*NFREQ).reshape(NFREQ, NTIME)
    else:
        data = background_noise

    # What about reading in noisy background?
    if sim is False:
        return data, []

    # Call class using parameter ranges
    ES = EventSimulator(dm=dm, scat_factor=scat_factor, fluence=fluence, 
                        width=width, spec_ind=spec_ind)
    # Realize event parameters for a single FRB
    dm, fluence, width, spec_ind, disp_ind, scat_factor = ES.draw_event_parameters()
    # Create event class with those parameters 
    E = Event(t_ref, FREQ_REF, dm, 10e-4*fluence, 
              width, spec_ind, disp_ind, scat_factor)
    # Add FRB to data array 
    E.add_to_data(delta_t, freq, data)

    if plot_burst:
        subplot(211)
        imshow(data.reshape(-1, NTIME), aspect='auto', 
               interpolation='nearest', vmin=0, vmax=10)
        subplot(313)
        plot(data.reshape(-1, ntime).mean(0))

    return data, [dm, fluence, width, spec_ind, disp_ind, scat_factor]


def inject_in_filterbank(fn_fil, fn_fil_out, N_FRBs=1, NFREQ=1536, NTIME=2**15):
    """ Inject an FRB in each chunk of data 
        at random times. Default params are for Apertif data.
    """

    chunksize = 5e5
    ii=0

    params_full_arr = []

    for ii in xrange(N_FRBs):
        start, stop = chunksize*ii, chunksize*(ii+1)
        # drop FRB in random location in data chunk
        offset = int(np.random.uniform(10000, 400000)) 

        data, freq, delta_t, header = reader.read_fil_data(fn_fil, 
                                                start=start, stop=stop)

        if len(data[0])==0:
            break             

        data_event = (data[:NTIME].transpose()).astype(np.float)

        data_event, params = gen_simulated_frb(NFREQ=NFREQ, 
                            NTIME=NTIME, sim=True, fluence=(2), 
                            spec_ind=(-4, 4), width=(delta_t, 2), 
                            dm=(100, 1000), scat_factor=(-3, -0.5), 
                            background_noise=data_event, 
                            delta_t=delta_t, plot_burst=False, 
                            freq=(1550, 1250), 
                            FREQ_REF=1400.)

        params.append(offset)
        print("Injecting with DM:%f width: %f offset: %d" % 
                                (params[0], params[2], offset))
        print(ii, params)
        
        data[offset:offset+NTIME] = data_event.transpose()
        print(data.dtype)

        params_full_arr.append(params)

        if ii==0:
            fn_rfi_clean = reader.write_to_fil(data, header, fn_fil_out)
        elif ii>0:
            fil_obj = reader.filterbank.FilterbankFile(fn_fil_out, mode='readwrite')
            fil_obj.append_spectra(data) 

        del data 

    return params_full_arr

# a, p = gen_simulated_frb(NFREQ=1536, NTIME=2**15, sim=True, fluence=(2),
#                 spec_ind=(-4, 4), width=(dt), dm=(40.0),
#                 scat_factor=(-3, -0.5), background_noise=None, delta_t=dt,
#                 plot_burst=False, freq=(1550, 1250), FREQ_REF=1400., 
#                 )

# a, p = gen_simulated_frb(NFREQ=1536, NTIME=2**11, sim=True, fluence=(2),
#                 spec_ind=(-4, 4), width=(dt, 1), dm=(50, 60),
#                 scat_factor=(-3, -0.5), background_noise=None, delta_t=dt,
#                 plot_burst=False, freq=(800, 400), FREQ_REF=600., 
#                 )

fn_fil = '/data/09/filterbank/20171213/2017.12.13-21:13:51.B0531+21/CB21_injectedFRB.fil'
fn_fil_out = '/data/09/filterbank/20171213/2017.12.13-21:13:51.B0531+21/test.fil'
p = inject_in_filterbrank(fn_fil, fn_fil_out, N_FRBs=10, NFREQ=1536)
np.savetxt(fn_fil_out+'.params', p)

def run_full_simulation(sim_obj, tel_obj, mk_plot=False, 
                        fn_rfi='./data/all_RFI_8001.npy', 
                        ftype='hdf5', dm_time_array=True):

    outdir = './data/'
    outdir = '/drives/G/0/simulated/'
    outfn = outdir + "data_nt%d_nf%d_dm%d_snrmax%d.%s" \
                    % (sim_obj._NTIME, sim_obj._NFREQ, 
                       round(max(sim_obj._dm)), sim_obj._SNR_MAX, ftype)

    if fn_rfi is not None:
        data_rfi, y = sim_obj.get_false_positives(fn_rfi)
    else:
        data_rfi, y = sim_obj.generate_noise()

    print("\nUsing %d false-positive triggers" % sim_obj._NRFI)
    print("Simulating %d FRBs\n" % sim_obj._NSIM)

    # if data_rfi[0].shape != (sim_obj._NFREQ*sim_obj._NTIME,):
    #     data_rfi = np.random.normal(0, 1, 
    #                sim_obj._NRFI*sim_obj._NFREQ*sim_obj._NTIME)
    #     print("Using simulated noise")

    arr_sim_full = [] # data array with all events
    yfull = [] # label array FP=0, TP=1
    arr_dm_time_full = []

    params_full_arr = []
    width_full_arr = []

    snr = [] # Keep track of simulated FRB signal-to-noise
    ii = -1
    jj = 0

    # Hack
    f_noise = None #data_rfi[NRFI:].copy().reshape(-1, 16, 250)
    sim_obj._NSIM=1 #hack
    # Loop through total number of events
    while jj < (sim_obj._NRFI + sim_obj._NSIM):
        jj = len(arr_sim_full)
        ii += 1
        if ii % 500 == 0:
            print("simulated:%d kept:%d" % (ii, jj))

        # If ii is greater than the number of RFI events in f, 
        # simulate an FRB
        #sim = bool(ii >= NRFI)

        if ii < sim_obj._NRFI:
            data = data_rfi[ii].reshape(sim_obj._NFREQ, sim_obj._NTIME)

            # Normalize data to have unit variance and zero median
            data = reader.rebin_arr(data, sim_obj._NFREQ, sim_obj._NTIME)
            data = dataproc.normalize_data(data)

            arr_sim_full.append(data.reshape(sim_obj._NFREQ*sim_obj._NTIME)[None])
            yfull.append(0) # Label the RFI with '0'
            continue

        elif (ii >=sim_obj._NRFI and jj < (sim_obj._NRFI + sim_obj._NSIM)):
            if f_noise is not None:
                noise = (f_noise[jj-NRFI]).copy()
            else:
                noise = None

            # maybe should feed gen_sim a tel object and 
            # a set of burst parameters... 
            arr_sim, params = gen_simulated_frb(NFREQ=sim_obj._NFREQ, 
                                                NTIME=sim_obj._NTIME, 
                                                delta_t=tel_obj._DELTA_T, 
                                                freq=tel_obj._freq,
                                                FREQ_REF=tel_obj._FREQ_REF,
                                                spec_ind=sim_obj._spec_ind, 
                                                width=sim_obj._width, 
                                                scat_factor=sim_obj._scat_factor, 
                                                dm=sim_obj._dm,
                                                fluence=sim_obj._fluence,
                                                background_noise=noise, 
                                                plot_burst=False,
                                                sim=True,                                        
                                                )

            # Normalize data to have unit variance and zero median
            arr_sim = reader.rebin_arr(arr_sim, sim_obj._NFREQ, sim_obj._NTIME)
            arr_sim = dataproc.normalize_data(arr_sim)

            # get SNR of simulated pulse. Center should be at ntime//2
            # rebin until max SNR is found.
            snr_ = tools.calc_snr(arr_sim.mean(0))

            # Only use events within a range of signal-to-noise
            if snr_ > sim_obj._SNR_MIN and snr_ < sim_obj._SNR_MAX:
                arr_sim_full.append(arr_sim.reshape(-1, sim_obj._NFREQ*sim_obj._NTIME))
                yfull.append(1) # Label the simulated FRB with '1'
                params_full_arr.append(params) # Save parameters bursts
                snr.append(snr_)
                continue
            else:
                continue

    if dm_time_array is True:
        E = Event(0, tel_obj._FREQ_REF, 0.0, 1.0, tel_obj._DELTA_T, 0., )

        for ii, data in enumerate(arr_sim_full):
            if ii%500==0:
                print("DM-transformed:%d" % ii)

            data = data.reshape(-1, sim_obj._NTIME)
            data = dataproc.normalize_data(data)
            data_dm_time = E.dm_transform(tel_obj._DELTA_T, data, tel_obj._freq)
            data_dm_time = dataproc.normalize_data(data_dm_time)
            arr_dm_time_full.append(data_dm_time)

        NDM = data_dm_time.shape[0]
        arr_dm_time_full = np.concatenate(arr_dm_time_full)
        arr_dm_time_full = arr_dm_time_full.reshape(-1, NDM, sim_obj._NTIME)
    else:
        data_dm_time_full = None

    params_full_arr = np.concatenate(params_full_arr).reshape(-1, 6)
    snr = np.array(snr) 
    yfull = np.array(yfull)
    
    arr_sim_full = np.concatenate(arr_sim_full, axis=-1)
    arr_sim_full = arr_sim_full.reshape(-1, sim_obj._NFREQ*sim_obj._NTIME)

    print("\nGenerated %d simulated FRBs with mean SNR: %f" 
                            % (sim_obj._NSIM, snr.mean()))
    print("Used %d RFI triggers" % sim_obj._NRFI)
    print("Total triggers with SNR>10: %d" % arr_sim_full.shape[0])

    if ftype is 'hdf5':
        arr_sim_full = arr_sim_full.reshape(-1, sim_obj._NFREQ, sim_obj._NTIME)
        sim_obj.write_sim_data(arr_sim_full, yfull, outfn, 
                               data_dm_time=arr_dm_time_full,
                               params=params_full_arr, 
                               snr=snr)
    else:
        full_label_arr = np.concatenate((arr_sim_full, yfull[:, None]), axis=-1)
        print("Saving training/label data to:\n%s" % outfn)

        # save down the training data with labels
        np.save(outfn, full_label_arr)

    if plt==None:
        mk_plot = False 

    if sim_obj._mk_plot == True:
        figname = './plots/training_set'
        kk=0

        plot_tools.plot_simulated_events(
                arr_sim_full, y, figname, 
                sim_obj._NSIDE, sim_obj._NFREQ, 
                sim_obj._NTIME, cmap='Greys')

    return arr_sim_full, yfull, params_full_arr, snr 




