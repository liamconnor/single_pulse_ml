import unittest
import numpy as np

from single_pulse_ml import simulate_frb

class TestSimulate_FRB(unittest.TestCase):

	def test_gen_simulated_frb(self):


		sim_data, params = simulate_frb.gen_simulated_frb(NFREQ=16, NTIME=250, sim=True, 
				   fluence=(0.03,0.3),
                   spec_ind=(-4, 4), width=(2*0.0016, 1), dm=(-0.15, 0.15),
                   scat_factor=(-3, -0.5), background_noise=None, delta_t=0.0016,
                   plot_burst=False, freq=(800, 400), FREQ_REF=600., 
                   )

		dm, fluence, width, spec_ind, disp_ind, scat_factor = params 

		print(dm)
		assert np.abs(dm) < 0.2, "DM is not in correct DM range"
		assert width > 0, "Width must be positive"
		assert disp_ind==2, "Disp index doesn't match input"


if __name__ == '__main__':
    unittest.main()
