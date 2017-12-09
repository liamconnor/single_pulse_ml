# Hacked together a script for simulating
# multi-beam detections 
# 5 December 2017 
# Liam Connor
import sys

import numpy as np
from numpy.random import seed
import h5py

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Merge
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPooling2D, MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.models import load_model

import frb_keras_convnet

def gauss(x, xo, sig):
    return np.exp(-(x-xo)**2/sig**2)

def generate_multibeam(nbeam=32, width=27, n=1000):
	""" width in arcminutes
	"""
	width /= 60.
	x = np.linspace(-1, 1, 100)
	ff = gauss(x, 0, width)
	a = ff[None]*ff[:, None]
	beam_arr = np.zeros([n, n, nbeam])

	# Make each beam 
	kk=0
	for ii in range(8):
	    for jj in range(4):
	        xx, yy = 500-4*50+ii*50, 500-2*50+jj*50
	        beam_arr[xx:xx+100, yy:yy+100, kk] += a
	        kk+=1

	return beam_arr

def make_dataset(ntrigger=2304, tp_frac=0.5):
	A = generate_multibeam()
	# Take a euclidean flux distribution
	sn = np.random.uniform(1, 1000, 10000)**-(2/3.) 
	sn /= np.median(sn)
	sn *= 15
	#sn[sn > 150] = 150

	det_ = []
	sn_ = []
	multis = 0

	for ii, ss in enumerate(sn):
	    xi = np.random.uniform(400, 650)
	    yi = np.random.uniform(300, 750)
	    abeams = A[int(xi), int(yi)] * ss
	    beamdet = np.where(abeams>=7)[0]
	    if len(beamdet)>0:
	        det_.append(beamdet)
	        sn_.append(abeams[beamdet])
	        if len(beamdet)>1:
	         	multis += 1

	# Check if fraction of multibeam detections is expected
	print np.float(multis) / len(det_)

	ntrigger = min(2*len(det_), ntrigger)
	nbeam = 32 # number of beams
	#data = np.random.normal(0, 1, n*m).reshape(-1, m)
	data = np.zeros([nbeam*ntrigger]).reshape(-1, nbeam)
	N_FP = int((1-tp_frac)*ntrigger)
	N_TP = int(tp_frac*ntrigger)

	for ii in range(N_FP):
	    nbeam = int(np.random.uniform(1, 32))
	    ind = set(np.random.uniform(1, 32, nbeam).astype(int).astype(list))
	    data[ii][list(ind)] = np.random.normal(20, 5, len(ind))

	for ii in range(N_TP):
	#    beam = int(np.random.uniform(1, 32))
	    data[N_FP+ii][det_[ii]] = sn_[ii]#np.random.normal(20, 5, 1)

	labels = np.zeros([ntrigger])
	labels[N_FP:] = 1 
	labels = labels.astype(int)
	labels = keras.utils.to_categorical(labels)

	return data, labels
