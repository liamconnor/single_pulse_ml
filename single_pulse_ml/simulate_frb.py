import random
import math
import uuid
import logging

import numpy as np
import numpy.random as nprand

import reader
import dataproc

class Event(object):

    def __init__(self, t_ref, f_ref, dm, fluence, width, spec_ind, disp_ind, scat_factor=0):
        self._t_ref = t_ref
        self._f_ref = f_ref
        self._dm = dm
        self._fluence = fluence 
        self._width = width
        self._spec_ind = spec_ind
        self._disp_ind = disp_ind
        self._scat_factor = scat_factor

    def disp_delay(f, _dm, _disp_ind):
        return 4.148808e3 * _dm * (f**(-2))

    def arrival_time(self, f):
        t = disp_delay(f, self._dm, self._disp_ind)
        t = t - disp_delay(self._f_ref, self._dm, self._disp_ind)
        return self._t_ref + t

    # Liam DM smearing 
    def dm_smear(self, DM, freq_c, delta_freq=400.0/1024, ti=1e3, tsamp=2.56*512, tau=5e3):  
        """ Calculate DM smearing SNR reduction
        """
        tI = np.sqrt(ti**2 + tsamp**2 + (8.3 * DM * delta_freq / freq_c**3)**2)

        return (np.sqrt(ti**2 + tau**2) / tI)**0.5

    def add_to_data(self, t0, delta_t, freq, data, dm=0.0):
        ntime = data.shape[1]

        for ii, f in enumerate(freq):
            t = self.arrival_time(f)
            start_ind = (np.round((t - t0) / delta_t)).astype(int)
            width = self._width + self._scat_factor * self._width * (f / self._f_ref)**-4
            stop_ind = (np.round((t + width - t0) / delta_t)).astype(int)
            start_ind = max(0, start_ind)
            start_ind = min(ntime, start_ind)
            stop_ind = max(0, stop_ind)
            stop_ind = min(ntime, stop_ind)
            val = self._fluence / self._width * np.std(data[ii])
#            val = 50.0 * np.std(data[ii]) #* self.dm_smear(dm, f/1000.0)#liam set to small snr per bin
            val = val * (f / self._f_ref) ** self._spec_ind 
            data[ii,start_ind:stop_ind] += val

    def add_to_data_dedispersed(self, td, delta_t, freq, data):
        """ td should be the offset from the central pixel in seconds
        """

        ntime = data.shape[1]
        tmid = ntime//2

        for ii, f in enumerate(freq):
            start_ind = tmid + int(round(td/delta_t))
            width = self._width + self._scat_factor * self._width * (f / self._f_ref)**-4
            stop_ind = start_ind + (np.round((width/ delta_t))).astype(int)
            start_ind = max(0, start_ind)
            start_ind = min(ntime, start_ind)
            stop_ind = max(0, stop_ind)
            stop_ind = min(ntime, stop_ind)
            val = self._fluence / self._width
#            val = 50.0 * np.std(data[ii]) #* self.dm_smear(dm, f/1000.0)#liam set to small snr per bin
            val = val * (f / self._f_ref) ** self._spec_ind 
            data[ii,start_ind:stop_ind] += val



class EventSimulator():
    """Generates simulated fast radio bursts.

    Events occurances are drawn from a poissonian distribution.

    The events are rectangular in time with properties each evenly distributed
    within a range that is determined at instantiation.

    """


    def __init__(self, rate=0.001, dm=(0.,2000.), fluence=(0.03,0.3),
                 width=(0.001, 0.010), spec_ind=(-4.,4), disp_ind=2., scat_factor=(0, 0.5)):
        """

        Parameters
        ----------
        datasource : datasource.DataSource object
            Source of the data, specifying the data rate and band parameters.
        rate : float
            The average rate at which to simulate events (per second).
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

        """

        self._rate = rate
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

        self._freq = np.linspace(800, 400, 16)
        self._delta_t = 0.0016

        self._simulated_events = []

        self._last_time_processed = 0.

    def draw_event_parameters(self):
        dm = uniform_range(*self._dm)
        fluence = 10*uniform_range(*self._fluence)**(-2/3.) / 0.5**(-2/3.)
        width = uniform_range(*self._width)
        spec_ind = uniform_range(*self._spec_ind)
        disp_ind = uniform_range(*self._disp_ind)
        scat_factor = uniform_range(*self._scat_factor)
        return dm, fluence, width, spec_ind, disp_ind, scat_factor

    def inject_events(self, t0, data):
        """Assumes that subsequent calls always happen with later t0,
        although blocks may overlap."""

        ntime = data.shape[1]
        time = np.arange(ntime) * self._delta_t + t0
        f_max = max(self._freq)
        f_min = min(self._freq)
        f_mean = np.mean(self._freq)

        overlap_events = [e for e in self._simulated_events if
                          e.arrival_time(f_min) > t0]

        new_events = []
        mu = self._rate * self._delta_t
        # Liam hardcode the rate to be 1/50 blocks
        mu = .02 * ntime**-1
        mu = ntime**-1

        events_per_bin = nprand.poisson(mu, ntime)
        
        events_per_bin *= 0 #liam plopping a simulated even at middle pixel
        events_per_bin[ntime//2] = 1

        events_per_bin[time <= self._last_time_processed] = 0
        event_inds, = np.where(events_per_bin)
        for ind in event_inds:
            for ii in range(events_per_bin[ind]):
                dm, fluence, width, spec_ind, disp_ind, scat_factor = \
                        self.draw_event_parameters()
                
                scat_factor = 0.0 # liam test
                spec_ind = 0.0
                width = 0.005
#                fluence = 100.0

                msg = ("Injecting simulated event at time = %5.2f, DM = %6.1f,"
                       " fluence = %f, width = %f, spec_ind = %3.1f, disp_ind"
                       " = %3.1f., disp_ind = %3.1f.")
                logger.info(msg
                        % (time[ind], dm, fluence, width, spec_ind, disp_ind, scat_factor))
                t = disp_delay(f_min, dm, disp_ind)
                t = t - disp_delay(f_mean, dm, disp_ind)
                t = t + time[ind]
                e = Event(t, f_mean, dm, fluence, width, spec_ind, disp_ind, scat_factor)
                new_events.append(e)

        for e in overlap_events + new_events:
            #e.add_to_data(t0, self._delta_t, self._freq, data)
            e.add_to_data(t0, self._delta_t, self._freq, data, dm=dm) #liam

        self._simulated_events = self._simulated_events + new_events
        self._last_time_processed = time[-1]



def uniform_range(min_, max_):
    return random.uniform(min_, max_)

def gen_simulated_frb(nfreq=16, ntime=250, freq=np.linspace(800, 400, 16)):
    """ Simulate fast radio bursts using the EventSimulator class.

    Parameters
    ----------

    """
    plot = False

    # Hard code incoherent Pathfinder data time resolution
    t_ref, f_ref, delta_t = 0.2, 600., 0.0016    

    # Generate background noise with unit variance
    data = np.random.normal(0, 1, ntime*nfreq).reshape(nfreq, ntime)

    ES = EventSimulator(dm=(-15, 15), scat_factor=(0,0.01), width=(0.001, 0.002))
    dm, fluence, width, spec_ind, disp_ind, scat_factor = ES.draw_event_parameters()
    E = Event(t_ref, f_ref, dm, 2e-4*fluence, width, spec_ind, disp_ind, scat_factor)
    E.add_to_data_dedispersed(0., delta_t, freq, data)

    if plot:
        subplot(211)
        imshow(data.reshape(-1, 250), aspect='auto', interpolation='nearest', vmin=0, vmax=10)
        subplot(313)
        plot(data.reshape(-1, 250).mean(0))

    data = reader.rebin_arr(data, nfreq, ntime)
    data = dataproc.normalize_data(data)#[astart:aend, :]

    return data

if __name__=='__main__':
    #f = np.load('/Users/connor/code/machine_learning/single_pulse_ml/single_pulse_ml/data/pathfinder_all_sim.npy')
    #f = np.load('/Users/connor/code/machine_learning/single_pulse_ml/single_pulse_ml/data/allRFI_i_think.npy')

    # Read in false positive triggers from the Pathfinder
    f = np.load('/Users/connor/code/machine_learning/single_pulse_ml/single_pulse_ml/data/all_RFI_8001.npy')
    nfreq, ntime = 16, 250
    freq = np.linspace(800, 400, 16)
    mk_plot = False

    # Read in data array and labels from RFI file
    d, y = f[:, :-1], f[:, -1]

    # simulate one FRB for each RFI trigger
    nsim = len(f)
    nrfi = len(d)

    Afull = []
    snr = [] # Keep track of simulated FRB signal-to-noise
    yfull = []

    for ii in xrange(nrfi + nsim):
        sim = bool(ii >= len(d))
        if ii >= len(d):
            A = gen_simulated_frb(freq=freq)

            # get SNR of simulated pulse. Center should be at ntime//2
            snr_ = A.mean(0).max() / np.std(A.mean(0)[:ntime//4])
            snr.append(snr_)

            # for now, use a SNR cutoff of 10
            if snr_ > 10.0:
                Afull.append(A.reshape(-1, nfreq*ntime))
                yfull.append(int(sim))
                continue
            else:
                continue

        elif ii < len(d): 
            Afull.append(d[ii].reshape(-1, nfreq*ntime))
            yfull.append(y[ii])

    snr = np.array(snr)
    yfull = np.array(yfull)
    Afull = np.concatenate(Afull, axis=-1).reshape(-1, nfreq*ntime)

    print "Generated %d simulated FRBs with mean SNR: %d" % (nsim, snr.mean())
    print "Used %d RFI triggers" % nrfi

    outfn = '/Users/connor/code/machine_learning/single_pulse_ml/single_pulse_ml/data/test_triggers.npy'
    full_label_arr = np.concatenate((Afull, yfull[:, None]), axis=-1)

    print "Saving training/label data to:\n%s" % outfn
    # save down the training data with labels
    np.save(outfn, full_label_arr)

    if mk_plot is True:
        kk=0
        for ii in range(100):
            subplot(10,10,ii+1)
            imshow(Afull[ii+kk].reshape(-1, 250), aspect='auto', interpolation='nearest')
            axis('off')
            title(str(y[ii+kk]))

        figure()

        for ii in range(100):
            subplot(10,10,ii+1)
            plot(Afull[ii+kk].reshape(-1, 250).mean(0))
            title(str(y[ii+kk]))






