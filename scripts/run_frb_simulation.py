#! /usr/bin/python
""" Script to build dataset out of simulated 

single pulses + false positive triggers
"""

from single_pulse_ml import sim_parameters
from single_pulse_ml import telescope
from single_pulse_ml import simulate_frb

# TELESCOPE PARAMETERS:
freq = (800, 400)   # (FREQ_LOW, FREQ_UP) in MHz
FREQ_REF = 600      # dispersion reference frequency in MHz
DELTA_T = 0.0016    # time resolution in seconds
NAME = "CHIMEPathfinder"

# SIMULATION PARAMETERS 
NFREQ = 32  # Must agree with false-positive data
NTIME = 250 
dm = (-0.1, 0.1)
fluence = 100#(0.005, 0.5)
width = (2*0.0016, 0.05) # width lognormal dist in seconds
spec_ind = (-4., 4.)
disp_ind = 2. 
scat_factor = (-4., -1.)
NRFI = 4650
SNR_MIN = 5.0
SNR_MAX = 50.0
out_file_name = None, 
mk_plot = True
NSIDE = 8
dm_time_array = False
outname_tag = 'test'

fn_rfi = './data/pathfinder_training_data/all_rfi_november17/data_rfi_shuffled.hdf5'
fn_rfi = None

sim_obj = sim_parameters.SimParams(dm=dm, fluence=fluence,
                                   width=width, spec_ind=spec_ind,
                                   disp_ind=disp_ind, scat_factor=scat_factor,
                                   SNR_MIN=SNR_MIN, SNR_MAX=SNR_MAX,
                                   out_file_name=out_file_name, NRFI=NRFI,
                                   NTIME=NTIME, NFREQ=NFREQ,
                                   mk_plot=mk_plot, NSIDE=NSIDE, )
                                   
tel_obj = telescope.Telescope(freq=freq, FREQ_REF=FREQ_REF,
                              DELTA_T=DELTA_T, name=NAME)

data, labels, params, snr = simulate_frb.run_full_simulation(
                                    sim_obj, tel_obj, fn_rfi=fn_rfi,
                                    dm_time_array=dm_time_array)
