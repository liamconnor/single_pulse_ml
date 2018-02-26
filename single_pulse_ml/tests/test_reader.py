import unittest
from unittest import TestCase
import h5py
import numpy as np

from single_pulse_ml import reader

class TestReader(TestCase):

	def test_read_hdf5(self):
		NFREQ = 64
		NTIME = 250
		NCANDIDATES = 100
		data_freq_time = np.random.normal(0, 1, NFREQ*NTIME*NCANDIDATES)
		data_freq_time.shape = (NCANDIDATES, NFREQ, NTIME)
		labels = np.ones([NCANDIDATES])
		fn = './test.hdf5'

		g = h5py.File(fn,'w')
		g.create_dataset('data_freq_time', data=data_freq_time)
		g.create_dataset('labels', data=labels)
		g.create_dataset('data_dm_time', data=[])				
		g.close()

		data, y = reader.read_hdf5(fn)
		self.assertTrue()


if __name__ == '__main__':
    unittest.main()
