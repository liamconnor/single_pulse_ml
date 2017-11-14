# Script to run FRB simulation

import sim_parameters
import telescope
import simulate_frb

# TELESCOPE PARAMETERS:
freq = (800, 400)   # (FREQ_LOW, FREQ_UP) in MHz
FREQ_REF = 600      # reference frequency in MHz
DELTA_T = 0.0016    # time res in seconds
NAME = "CHIMEPathfinder"

# SIMULATION PARAMETERS 
NFREQ = 32 # Number of frequencies. Must agree with FP data
NTIME = 250 # Number of time stamps per trigger
dm = (-0.01, .01)
fluence = (0.1, 0.3)
width = width=(3*0.0016, 0.75) # width lognormal dist in seconds
spec_ind=(-3., 3.)
disp_ind=2. 
scat_factor=(-4., -1.)
NRFI = 1000
SNR_MIN=10.
SNR_MAX=150.
out_file_name=None, 
mk_plot=True
NSIDE=8

fn_rfi = './data/pathfinder_training_data/all_rfi_november17/data_rfi_shuffled.hdf5'

sim_obj = sim_parameters.SimParams(dm=dm, fluence=fluence,
                 width=width, spec_ind=spec_ind,
                 disp_ind=disp_ind, scat_factor=scat_factor, 
                 SNR_MIN=SNR_MIN, SNR_MAX=SNR_MAX, 
                 out_file_name=out_file_name, NRFI=NRFI,
                 NTIME=NTIME, NFREQ=NFREQ, mk_plot=mk_plot, NSIDE=NSIDE)


tel_obj = telescope.Telescope(freq=freq, FREQ_REF=FREQ_REF, 
				    DELTA_T=DELTA_T, name=NAME)


data, labels, params, snr = simulate_frb.run_full_simulation(
												sim_obj, tel_obj, fn_rfi=fn_rfi)

