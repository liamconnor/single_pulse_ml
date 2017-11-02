import numpy as np 


class Telescope():

	def __init__(self, freq=(800, 400), FREQ_REF=600,
				 DELTA_T=0.0016, name=None):
		""" Telescope class that can be fed to simulation 

		Parameters:
		-----------
		freq : tuple 
			two-element tuple with (FREQ_LOW, FREQ_UP) in MHz
			e.g. for CHIME this is (800., 400.)
		DELTA_T : float
			time resolution in seconds 
		NFREQ : int 
			number of frequencies
		NTIME : int 
			number of time samples 
		name : str 
			telescope name, e.g. CHIME_PATHFINDER 

		"""
		self._FREQ_LOW = freq[0]
		self._FREQ_UP = freq[-1]
		self._freq = freq 
		self._FREQ_REF = FREQ_REF
		self._DELTA_T = DELTA_T
		self._telname = name
