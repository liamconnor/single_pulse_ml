# single_pulse_ml

Build, train, and apply deep neural networks to single pulse candidates. 

run_frb_simulation.py constructs a training set that includes simulated FRBs

run_single_pulse_DL.py allows for training of deep neural networks for several 
input data products, including:
  -- dedispersed dynamic specta (2D CNN)
  -- DM/time intensity array (2D CNN)
  -- frequency-collapsed pulse profile (1D CNN)
  -- Multi-beam S/N information (1D feed forward DNN)
  
run_single_pulse_DL.py can also be used when a trained model already exists and candidates are to be classified
