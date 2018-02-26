""" Test script generating 100 RFI events + 
100 simulated FRBs. Gaussian noise is used. 
"""

from single_pulse_ml import sim_parameters
from single_pulse_ml import telescope
from single_pulse_ml import simulate_frb

# TELESCOPE PARAMETERS:
freq = (800, 400)   # (FREQ_LOW, FREQ_UP) in MHz
FREQ_REF = 600      # reference frequency in MHz
DELTA_T = 0.0016    # time res in seconds
NAME = "CHIMEPathfinder"

# SIMULATION PARAMETERS 
NFREQ = 32  # Number of frequencies. Must agree with FP data
NTIME = 250 # Number of time stamps per trigger
dm = (-0.05, 0.05)
fluence = (5, 100)
width = (2*0.0016, 0.75) # width lognormal dist in seconds
spec_ind = (-4., 4.)
disp_ind = 2.
scat_factor = (-4., -1.5)
NRFI = 100
SNR_MIN = 8.0
SNR_MAX = 100.0
out_file_name = None, 
mk_plot = True
NSIDE = 8
dm_time_array = False
outname_tag = 'test'
outdir = '../data/'

fn_rfi = None
fn_noise = None

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
                                    fn_noise=fn_noise,
                                    dm_time_array=dm_time_array, 
                                    outname_tag=outname_tag, outdir=outdir)

