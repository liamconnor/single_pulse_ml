import unittest
import numpy as np

from single_pulse_ml import frbkeras 

class TestFRBkeras(unittest.TestCase):

	def test_get_classification_results(self):
		""" Test that the treu/false postives/negatives, 
		are correctly identified.
		"""
		y_true = np.round(np.random.rand(10000))
		y_pred = np.round(np.random.rand(10000))

		TP, FP, TN, FN = frbkeras.get_classification_results(y_true, y_pred)
		minlen = min(np.array([len(TP), len(FP), len(TN), len(FN)]))
		assert minlen>0, "There should be more than 0 of all"

		# Now create 1000 false events that are predicted true
		y_true = np.zeros([1000])
		y_pred = np.ones([1000])

		TP, FP, TN, FN = frbkeras.get_classification_results(y_true, y_pred)

		assert len(TP)==0
		assert len(FP)!=0
		assert len(TN)==0
		assert len(FN)==0

	def test_construct_conv2d(self):
		""" Test the 2d CNN by generating fake 
		data (gaussian noise) and fitting model
		""" 
		ntime = 64
		nfreq = 32
		ntrigger = 1000

		data = np.random.normal(0, 1, ntrigger*nfreq*ntime)
		data.shape = (ntrigger, nfreq, ntime, 1)
		labels = np.round(np.random.rand(ntrigger))
		labels = frbkeras.keras.utils.to_categorical(labels)

		# try training a model on random noise. should not do 
		# better than ~50% acc
		model, score = frbkeras.construct_conv2d(train_data=data[::2], 
												 train_labels=labels[::2],
												 eval_data=data[1::2],
												 eval_labels=labels[1::2],
												 fit=True, epochs=3)
		assert score[1]<0.9, "Trained on random noise. Should not have high acc"
		self.model_conv2d = model 


	def test_construct_conv1d(self):
		""" Test the 1d CNN by generating fake 
		data (gaussian noise) and fitting model
		""" 
		ntime = 64
		ntrigger = 1000

		data = np.random.normal(0, 1, ntrigger*ntime)
		data.shape = (ntrigger, ntime, 1)
		labels = np.round(np.random.rand(ntrigger))
		labels = frbkeras.keras.utils.to_categorical(labels)

		# try training a model on random noise. should not do 
		# better than ~50% acc
		model, score = frbkeras.construct_conv1d(fit=True, train_data=data[::2], 
												 train_labels=labels[::2],
												 eval_data=data[1::2],
												 eval_labels=labels[1::2],
												 batch_size=16, epochs=3)

		assert score[1]<0.9, "Trained on random noise. Should not have high acc"

		self.model_conv1d = model 

	def test_construct_ff1d(self):
		nbeam = 32
		ntrigger = 1000

		data = np.random.normal(0, 1, ntrigger*nbeam)
		data.shape = (ntrigger, nbeam, 1)
		labels = np.round(np.random.rand(ntrigger))
		labels = frbkeras.keras.utils.to_categorical(labels)

		# try training a model on random noise. should not do 
		# better than ~50% acc
		model, score = frbkeras.construct_conv1d(fit=True, train_data=data[::2], 
												 train_labels=labels[::2],
												 eval_data=data[1::2],
												 eval_labels=labels[1::2],
												 batch_size=16, epochs=3)


if __name__ == '__main__':
    unittest.main()







