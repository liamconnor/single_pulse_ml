import random
import logging

import numpy as np
import numpy.random as nprand
import glob

try:
    import matplotlib.pyplot as plt
except:
    plt = None
    pass

import reader
import dataproc
import tools 
import plot_tools

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
                 spec_ind, disp_ind, scat_factor=0):
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

    def dm_smear(self, DM, freq_c, delta_freq=400.0/1024, 
                 ti=1e3, tsamp=2.56*512, tau=5e3):  
        """ Calculate DM smearing SNR reduction
        """
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
        pulse_prof = np.convolve(gaus_prof, scat_prof, mode='full')[:nt]

        return pulse_prof


    def add_to_data(self, delta_t, freq, data):
        """ Method to add already-dedispersed pulse 
        to background noise data. Includes frequency-dependent 
        width (smearing, scattering, etc.) and amplitude 
        (scintillation, spectral index). 
        """

        NTIME = data.shape[1]
        tmid = NTIME//2

        scint_amp = self.scintillation(freq)

        for ii, f in enumerate(freq):
            index_width = max(1, (np.round((self._width/ delta_t))).astype(int))
            tpix = int(self.arrival_time(f) / delta_t)
            pp = self.pulse_profile(NTIME, index_width, f, 
                                    tau=self._scat_factor, t0=tpix)
            val = pp.copy()#[:len(pp)//NTIME * NTIME].reshape(NTIME, -1).mean(-1)
            val /= val.max()
            val *= self._fluence / self._width
            val = val * (f / self._f_ref) ** self._spec_ind 
            val = (0.5 + scint_amp[ii]) * val 
            data[ii] += val


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
        fluence = 3*uniform_range(*self._fluence)**(-2/3.) / 0.5**(-2/3.)
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
                plot_burst=False, freq=(800, 400), FREQ_REF=600.):
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

    data = reader.rebin_arr(data, NFREQ, NTIME)
    data = dataproc.normalize_data(data)

    return data, [dm, fluence, width, spec_ind, disp_ind, scat_factor]

def run_full_simulation(sim_obj, tel_obj, mk_plot=False):

    outdir = './data/'
    outfn = outdir + "_data_nt%d_nf%d_dm%d_snrmax%d.npy" \
                    % (sim_obj._NTIME, sim_obj._NFREQ, 
                       round(max(sim_obj._dm)), sim_obj._SNR_MAX)

    # need to figure out what to do with this one
    fn_rfi = './data/all_RFI_8001.npy'
    data_rfi, y = sim_obj.get_false_positives(fn_rfi)

    arr_sim_full = [] # data array with all events
    yfull = [] # label array FP=0, TP=1

    params_full_arr = []
    width_full_arr = []

    snr = [] # Keep track of simulated FRB signal-to-noise
    ii = 0
    jj = 0

    # Hack
    f_noise = None #data_rfi[NRFI:].copy().reshape(-1, 16, 250)

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
            data_rfi = np.random.normal(0, 1, sim_obj._NTIME*sim_obj._NFREQ)[None]
            arr_sim_full.append(data_rfi)
            # arr_sim_full.append(data_rfi[ii].reshape(-1, NFREQ*NTIME)) hack
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

            # get SNR of simulated pulse. Center should be at ntime//2
            # rebin until max SNR is found.
            snr_ = tools.calc_snr(arr_sim.mean(0))
            width_ii = params[2]
            # for now, reject events outside of some snr range
            if snr_ > sim_obj._SNR_MIN and snr_ < sim_obj._SNR_MAX:
                arr_sim_full.append(arr_sim.reshape(-1, sim_obj._NFREQ*sim_obj._NTIME))
                yfull.append(1) # Label the simulated FRB with '1'

                width_full_arr.append(width_ii)
                params_full_arr.append(params)
                snr.append(snr_)
                continue
            else:
                continue

    width_full_arr = np.array(width_full_arr)
    params_full_arr = np.concatenate(params_full_arr)
    snr = np.array(snr)
    yfull = np.array(yfull)

    arr_sim_full = np.concatenate(arr_sim_full, axis=-1)
    arr_sim_full = arr_sim_full.reshape(-1, sim_obj._NFREQ*sim_obj._NTIME)

    print("\nGenerated %d simulated FRBs with mean SNR: %f" % (sim_obj._NSIM, snr.mean()))
    print("Used %d RFI triggers" % sim_obj._NRFI)
    print("Total triggers with SNR>10: %d" % arr_sim_full.shape[0])

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

if __name__=='__main__':

    FREQ_LOW = 800. #telparam # first freq in array in MHz
    FREQ_REF = 600.
    FREQ_UP = 400. #telparam # last freq in array in MHz
    NFREQ = 16 #telparam
    NTIME = 250 #telparam
    freq = np.linspace(FREQ_LOW, FREQ_UP, NFREQ) #telparam

    DELTA_T = 0.0016 #telparam # Time resolution in seconds

    dm=(-.01, 0.01) # bparam
    fluence=(0.1,0.3) # bparam
    width=(3*0.0016, 0.75) # bparam
    spec_ind=(-3., 3.)  # bparam
    disp_ind=2. # bparam
    scat_factor=(-4., -1.) # bparam # hack

    SNR_MIN = 8.0 # bparam
    SNR_MAX = 100.0 # bparam

    # Let's try the dispersion version:
    # dm = (1.0, 25)
    # scat_factor = (-4.5)
    # width = (log(5*0.0016), 0.1)
    # spec_ind = (0.)
    # ntime = 1000
    # fluence=(0.03,0.3)*5

    # Plotting parameters
    mk_plot = True # sim param
    NSIDE = 7
    NFIG = NSIDE**2

    # Read in false positive triggers from the Pathfinder
    fn_rfi = './data/all_RFI_8001.npy'
    f_rfi = np.load(fn_rfi)
    #f_rfi = np.random.normal(0, 1, 2500*NTIME*NFREQ).reshape(-1, NTIME*NFREQ)

    # # Read in background data randomly selected and dedispersed
    # fn_noise = './data/background_pf_data.npy'
    # f_noise = np.load(fn_noise)
    # f_noise.shape = (-1, NFREQ, NTIME)

    outdir = './data/'
    outfn = outdir + "_data_nt%d_nf%d_dm%d_snrmax%d.npy" \
                    % (NTIME, NFREQ, round(max(dm)), SNR_MAX)
    figname = './plots/training_set' 

    # Important step! Need to scramble RFI triggers. 
    np.random.shuffle(f_rfi)

    # Read in data array and labels from RFI file
    data_rfi, y = f_rfi[:, :-1], f_rfi[:, -1]

    # simulate two FRBs for each RFI trigger
    NRFI = len(f_rfi)
    NSIM = NRFI

    arr_sim_full = []
    snr = [] # Keep track of simulated FRB signal-to-noise
    yfull = []
    ww_ = []
    ii = 0
    jj = 0

    # Hack
    f_noise = None#data_rfi[NRFI:].copy().reshape(-1, 16, 250)

    # Loop through total number of events
    while jj < (NRFI + NSIM):
        jj = len(arr_sim_full)
        ii += 1
        if ii % 500 == 0:
            print("simulated:%d kept:%d" % (ii, jj))

        # If ii is greater than the number of RFI events in f, 
        # simulate an FRB
        #sim = bool(ii >= NRFI)

        if ii < NRFI:
            data_rfi = np.random.normal(0, 1, NTIME*NFREQ)[None]
            arr_sim_full.append(data_rfi)
            # arr_sim_full.append(data_rfi[ii].reshape(-1, NFREQ*NTIME)) hack
            yfull.append(0) # Label the RFI with '0'
            continue

        elif (ii >=NRFI and jj < (NRFI + NSIM)):
            if f_noise is not None:
                noise = (f_noise[jj-NRFI]).copy()
            else:
                noise = None

            # maybe should feed gen_sim a tel object and 
            # a set of burst parameters... 
            arr_sim, params = gen_simulated_frb(NFREQ=NFREQ, NTIME=NTIME, sim=True, \
                        spec_ind=spec_ind, width=width, delta_t=DELTA_T, scat_factor=scat_factor, \
                        background_noise=noise, freq=(FREQ_LOW, FREQ_UP), FREQ_REF=FREQ_REF,\
                        plot_burst=False, dm=dm, fluence=fluence)

            # get SNR of simulated pulse. Center should be at ntime//2
            # rebin until max SNR is found.
            snr_ = tools.calc_snr(arr_sim.mean(0))
            ww = params[2]
            # for now, reject events outside of some snr range
            if snr_ > SNR_MIN and snr_ < SNR_MAX:
                arr_sim_full.append(arr_sim.reshape(-1, NFREQ*NTIME))
                yfull.append(1) # Label the simulated FRB with '1'

                ww_.append(ww)
                snr.append(snr_)
                continue
            else:
                continue

    ww_ = np.array(ww_)
    snr = np.array(snr)
    yfull = np.array(yfull)
    arr_sim_full = np.concatenate(arr_sim_full, axis=-1)
    arr_sim_full = arr_sim_full.reshape(-1, NFREQ*NTIME)

    print("\nGenerated %d simulated FRBs with mean SNR: %f" % (NSIM, snr.mean()))
    print("Used %d RFI triggers" % NRFI)
    print("Total triggers with SNR>10: %d" % arr_sim_full.shape[0])

    full_label_arr = np.concatenate((arr_sim_full, yfull[:, None]), axis=-1)

    print("Saving training/label data to:\n%s" % outfn)

    # save down the training data with labels
    np.save(outfn, full_label_arr)

    if plt==None:
        mk_plot = False 

    if mk_plot == True:
        kk=0

        plot_tools.plot_simulated_events(
                arr_sim_full, y, figname, 
                NSIDE, NFREQ, NTIME, cmap='Greys')






