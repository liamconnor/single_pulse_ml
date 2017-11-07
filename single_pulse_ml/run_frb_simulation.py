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
NFREQ = 128 # Number of frequencies. Must agree with FP data
NTIME = 250 # Number of time stamps per trigger
dm = (-0.01, .01)
fluence = (0.1, 0.3)
width = width=(3*0.0016, 0.75) # width lognormal dist in seconds
spec_ind=(-3., 3.)
disp_ind=2. 
scat_factor=(-4., -1.)
SNR_MIN=10.
SNR_MAX=100.
out_file_name=None, 
mk_plot=True
NSIDE=8

sim_obj = sim_parameters.SimParams(dm=dm, fluence=fluence,
                 width=width, spec_ind=spec_ind,
                 disp_ind=disp_ind, scat_factor=scat_factor, 
                 SNR_MIN=SNR_MIN, SNR_MAX=SNR_MAX, 
                 out_file_name=out_file_name,
                 NTIME=NTIME, NFREQ=NFREQ, mk_plot=mk_plot, NSIDE=NSIDE)


tel_obj = telescope.Telescope(freq=freq, FREQ_REF=FREQ_REF, 
				    DELTA_T=DELTA_T, name=NAME)


data, labels, params, snr = simulate_frb.run_full_simulation(
												sim_obj, tel_obj)

    # # Read in false positive triggers from the Pathfinder
    # fn_rfi = './data/all_RFI_8001.npy'
    # f_rfi = np.load(fn_rfi)
    # #f_rfi = np.random.normal(0, 1, 2500*NTIME*NFREQ).reshape(-1, NTIME*NFREQ)

    # # # Read in background data randomly selected and dedispersed
    # # fn_noise = './data/background_pf_data.npy'
    # # f_noise = np.load(fn_noise)
    # # f_noise.shape = (-1, NFREQ, NTIME)

    # outdir = './data/'
    # outfn = outdir + "_data_nt%d_nf%d_dm%d_snrmax%d.npy" \
    #                 % (NTIME, NFREQ, round(max(dm)), SNR_MAX)
    # figname = './plots/training_set' 

    # # Important step! Need to scramble RFI triggers. 
    # np.random.shuffle(f_rfi)

    # # Read in data array and labels from RFI file
    # data_rfi, y = f_rfi[:, :-1], f_rfi[:, -1]

    # # simulate two FRBs for each RFI trigger
    # NRFI = len(f_rfi)
    # NSIM = NRFI

    # arr_sim_full = []
    # snr = [] # Keep track of simulated FRB signal-to-noise
    # yfull = []
    # ww_ = []
    # ii = 0
    # jj = 0

    # # Hack
    # f_noise = None#data_rfi[NRFI:].copy().reshape(-1, 16, 250)


