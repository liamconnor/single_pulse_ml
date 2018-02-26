### single_pulse_ml

Build, train, and apply deep neural networks to single pulse candidates. 

run_frb_simulation.py constructs a training set that includes simulated FRBs

run_single_pulse_DL.py allows for training of deep neural networks for several 
input data products, including:
  -- dedispersed dynamic spectra (2D CNN)
  -- DM/time intensity array (2D CNN)
  -- frequency-collapsed pulse profile (1D CNN)
  -- Multi-beam S/N information (1D feed forward DNN)
  
run_single_pulse_DL.py can also be used when a trained model already exists and candidates are to be classified

This code has been used on CHIME Pathfinder incoherent data as well as commissioning data on Apertif. 

### Requirements

- You will need the following:
	- numpy 
	- scipy
	- h5py
	- matplotlib
	- tensorflow
	- keras

### Tests

In the single_pulse_ml/tests/ directory, 
"test_run_frb_simulation.py" can be run to generate 100 simulated FRBs
to ensure the simulation backend works.

"test_frbkeras.py" will generate 1000 gaussian-noise 
dynamic spectrum candidates of dimension 32x64, then
build, train, and test a CNN using the tools in frbkeras. 
This allows a test of the keras/tensorflow code.
