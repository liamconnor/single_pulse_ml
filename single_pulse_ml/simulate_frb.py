import random

import numpy as np
import glob
from scipy import signal

try:
    import matplotlib.pyplot as plt
except:
    plt = None
    pass

from single_pulse_ml import reader
from single_pulse_ml import dataproc
from single_pulse_ml import tools 

try:
    from single_pulse_ml import plot_tools
except:
    plot_tools = None


class Event(object):
    """ Class to generate a realistic fast radio burst and 
    add the event to data, including scintillation, temporal 
    scattering, spectral index variation, and DM smearing. 

    This class was expanded from real-time FRB injection 
    in Kiyoshi Masui's 
    https://github.com/kiyo-masui/burst\_search
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
        """ Calculated effective width of pulse 
        including DM smearing, sample time, etc.
        """

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
        f = np.linspace(0, 1, len(freq))

        # Make number of scintils between 0 and 10 (ish)
        nscint = np.exp(np.random.uniform(np.log(1e-3), np.log(7))) 
        #nscint=5
#        envelope = np.cos(nscint*(freq - self._f_ref)/self._f_ref + scint_phi)
        envelope = np.cos(2*np.pi*nscint*f + scint_phi)
        envelope[envelope<0] = 0

        return envelope

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
        rollind = 0#*int(np.random.normal(0, 5)) #hack
        
        for ii, f in enumerate(freq):
            width_ = 1e-3 * self.calc_width(self._dm, self._f_ref*1e-3, 
                                            bw=400.0, NFREQ=NFREQ,
                                            ti=self._width, tsamp=delta_t, tau=0)

#            width_ = self.dm_smear(self._dm, self._f_ref, 
#                                   delta_freq=400.0/1024, 
#                                   ti=self._width, tsamp=delta_t, tau=0)
            index_width = max(1, (np.round((width_/ delta_t))).astype(int))
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
            val = (0.1 + scint_amp[ii]) * val 
            val = np.roll(val, rollind)
            data[ii] += val

    def dm_transform(self, delta_t, data, freq, maxdm=5.0, NDM=50):
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


    This class was expanded from real-time FRB injection 
    in Kiyoshi Masui's 
    https://github.com/kiyo-masui/burst\_search
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
            fluence = (fluence[1]**-1, fluence[0]**-1)
            self._fluence = tuple(fluence)
        else:
            self._fluence = (float(fluence)**-1, float(fluence)**-1)
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


def inject_in_filterbank_background(fn_fil):
    """ Inject an FRB in each chunk of data 
        at random times. Default params are for Apertif data.
    """

    chunksize = 5e5
    ii=0

    data_full =[]
    nchunks = 250
    nfrb_chunk = 8
    chunksize = 2**16

    for ii in range(nchunks):
        downsamp = 2**((np.random.rand(nfrb_chunk)*6).astype(int))

        try:
            # drop FRB in random location in data chunk
            rawdatafile = filterbank.filterbank(fn_fil)
            dt = rawdatafile.header['tsamp']
            freq_up = rawdatafile.header['fch1']
            nfreq = rawdatafile.header['nchans']
            freq_low = freq_up + nfreq*rawdatafile.header['foff']
            data = rawdatafile.get_spectra(ii*chunksize, chunksize)
        except:
            continue
    

        #dms = np.random.uniform(50, 750, nfrb_chunk)
        dm0 = np.random.uniform(90, 750)
        end_width = abs(4e3 * dm0 * (freq_up**-2 - freq_low**-2))
        data.dedisperse(dm0)
        NFREQ, NT = data.data.shape

        print("Chunk %d with DM=%.1f" % (ii, dm0))
        for jj in xrange(nfrb_chunk):
            if 8192*(jj+1) > (NT - end_width):
                print("Skipping at ", 8192*(jj+1))
                continue
            data_event = data.data[:, jj*8192:(jj+1)*8192]
            data_event = data_event.reshape(NFREQ, -1, downsamp[jj]).mean(-1)
            print(data_event.shape)
            data_event = data_event.reshape(32, 48, -1).mean(1)

            NTIME = data_event.shape[-1]
            data_event = data_event[..., NTIME//2-125:NTIME//2+125]
            data_event -= np.mean(data_event, axis=-1, keepdims=True)
            data_full.append(data_event)

    data_full = np.concatenate(data_full)
    data_full = data_full.reshape(-1, 32, 250)

    np.save('data_250.npy', data_full)


def inject_in_filterbank(fn_fil, fn_fil_out, N_FRBs=1, 
                         NFREQ=1536, NTIME=2**15):
    """ Inject an FRB in each chunk of data 
        at random times. Default params are for Apertif data.
    """

    chunksize = 5e5
    ii=0

    params_full_arr = []

    for ii in xrange(N_FRBs):
        start, stop = chunksize*ii, chunksize*(ii+1)
        # drop FRB in random location in data chunk
        offset = int(np.random.uniform(0.1*chunksize, 0.9*chunksize)) 

        data, freq, delta_t, header = reader.read_fil_data(fn_fil, 
                                                start=start, stop=stop)

        # injected pulse time in seconds since start of file
        t0_ind = offset+NTIME//2+chunksize*ii
        t0 = t0_ind * delta_t

        if len(data[0])==0:
            break             

        data_event = (data[offset:offset+NTIME].transpose()).astype(np.float)

        data_event, params = gen_simulated_frb(NFREQ=NFREQ, 
                                               NTIME=NTIME, sim=True, fluence=(0.01, 1.), 
                                               spec_ind=(-4, 4), width=(delta_t, 2), 
                                               dm=(100, 1000), scat_factor=(-4, -0.5), 
                                               background_noise=data_event, 
                                               delta_t=delta_t, plot_burst=False, 
                                               freq=(1550, 1250), 
                                               FREQ_REF=1550.)

        params.append(offset)
        print("Injecting with DM:%f width: %f offset: %d" % 
                                (params[0], params[2], offset))
        
        data[offset:offset+NTIME] = data_event.transpose()

        #params_full_arr.append(params)
        width = params[2]
        downsamp = max(1, int(width/delta_t))

        params_full_arr.append([params[0], 20.0, t0, t0_ind, downsamp])

        if ii==0:
            fn_rfi_clean = reader.write_to_fil(data, header, fn_fil_out)
        elif ii>0:
            fil_obj = reader.filterbank.FilterbankFile(fn_fil_out, mode='readwrite')
            fil_obj.append_spectra(data) 

        del data 

    params_full_arr = np.array(params_full_arr)

    np.savetxt('/home/arts/connor/arts-analysis/simulated.singlepulse', params_full_arr)

    return params_full_arr

# a, p = gen_simulated_frb(NFREQ=1536, NTIME=2**15, sim=True, fluence=(2),
#                 spec_ind=(-4, 4), width=(dt), dm=(40.0),
#                 scat_factor=(-3, -0.5), background_noise=None, delta_t=dt,
#                 plot_burst=False, freq=(1550, 1250), FREQ_REF=1400., 
# #                 )

# a, p = gen_simulated_frb(NFREQ=32, NTIME=250, sim=True, fluence=(5, 100),
#                 spec_ind=(-4, 4), width=(dt, 1), dm=(-0.1, 0.1),
#                 scat_factor=(-3, -0.5), background_noise=None, delta_t=dt,
#                 plot_burst=False, freq=(800, 400), FREQ_REF=600., 
#                 )


def run_full_simulation(sim_obj, tel_obj, mk_plot=False, 
                        fn_rfi='./data/all_RFI_8001.npy',
                        fn_noise=None, 
                        ftype='hdf5', dm_time_array=True, 
                        outname_tag='', outdir = './data/'):

    outfn = outdir + "data_nt%d_nf%d_dm%d_snr%d-%d_%s.%s" \
                    % (sim_obj._NTIME, sim_obj._NFREQ, 
                       round(max(sim_obj._dm)), sim_obj._SNR_MIN, 
                       sim_obj._SNR_MAX, outname_tag, ftype)

    if fn_rfi is not None:
        data_rfi, y = sim_obj.get_false_positives(fn_rfi)
    else:
        data_rfi, y = sim_obj.generate_noise()

    if fn_noise is not None:
        noise_arr = np.load(fn_noise)   # Hack

    sim_obj._NRFI = min(sim_obj._NRFI, data_rfi.shape[0])
    print("\nUsing %d false-positive triggers" % sim_obj._NRFI)
    print("Simulating %d FRBs\n" % sim_obj._NSIM)

    arr_sim_full = [] # data array with all events
    yfull = [] # label array FP=0, TP=1
    arr_dm_time_full = []

    params_full_arr = []
    width_full_arr = []

    snr = [] # Keep track of simulated FRB signal-to-noise
    ii = -1
    jj = 0

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
            
            if fn_noise is not None:
                noise_ind = (jj-sim_obj._NRFI) % len(noise_arr) # allow for roll-over
                noise = (noise_arr[noise_ind]).copy()
                noise[noise!=noise] = 0.0
                noise -= np.median(noise, axis=-1)[..., None]
                noise -= np.median(noise)
                noise /= np.std(noise)
#                noise[:, 21] = 0 # hack mask out single bad channel
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
            snr_ = tools.calc_snr(arr_sim.mean(0), fast=False)

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
        print("Saving training/label data to:\n%s" % outfn)
    else:
        full_label_arr = np.concatenate((arr_sim_full, yfull[:, None]), axis=-1)
        print("Saving training/label data to:\n%s" % outfn)

        # save down the training data with labels
        np.save(outfn, full_label_arr)

    if plt==None:
        mk_plot = False 

    if sim_obj._mk_plot==True:
        figname = './plots/training_set'
        kk=0

        plot_tools.plot_simulated_events(
                arr_sim_full, y, figname, 
                sim_obj._NSIDE, sim_obj._NFREQ, 
                sim_obj._NTIME, cmap='Greys')

    return arr_sim_full, yfull, params_full_arr, snr 




