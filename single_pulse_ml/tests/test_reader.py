from unittest import TestCase

from single_pulse_ml import reader

class TestReader(TestCase):
	def test_create_training_set(self):
		a = reader.create_training_set()

	def test_read_training_set(self):
		fn = './single_pulse_ml/data/data_freqtime_train'
		data, y = reader.read_training_data(fn)
		self.assertTrue()