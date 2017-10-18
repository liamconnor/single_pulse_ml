""" Tools for io as well as creating training 
	and test data sets. 
"""

import os

import time
import numpy as np
import glob
import matplotlib.pylab as plt
import pickle


def read_pathfinder_npy(fn):
	data = np.load(fn)
	nfreq, ntimes = data.shape[0], data.shape[1]

	if len(data)!=16:
		data = data.reshape(-1, nfreq//16, ntimes).mean(1)

	return data

def rebin_arr(data, n0_f=1, n1_f=1):
	""" Rebin 2d array data to have shape 
		(n0_f, n1_f)
	"""
	assert len(data.shape)==2

	n0, n1 = data.shape
	data_rb = data[:n0/n0_f * n0_f, :n1/n1_f * n1_f]
	data_rb = data_rb.reshape(n0_f, n0/n0_f, n1_f, n1/n1_f)
	data_rb = data_rb.mean(1).mean(-1)
	return data_rb

def im(data, title='',figname='out.png'):
	fig = plt.figure()#
	plt.imshow(data, aspect='auto', interpolation='nearest', cmap='Greys')
#	plt.savefig(figname)
	plt.title(title)
	plt.show()


def combine_data_DT(fn):
	""" Combine the training set data in DM / Time space, 
	assuming text file with lines:

	# filepath label
	DM20-100_vdif_assembler+a=00+n=02_DM-T_ +11424.89s.npy 0
	DM20-100_vdif_assembler+a=00+n=02_DM-T_ +19422.29s.npy 1
	DM20-100_vdif_assembler+a=00+n=02_DM-T_ +21658.40s.npy 0

	e.g. usage: combine_data_DT('./single_pulse_ml/data/test/data_list_DM.txt')
	"""

	f = open(fn,'r')

	data_full, y = [], []
	k=0
	for ff in f:
		fn = './single_pulse_ml/data/' + ff.strip()[:-2]
		try:
			data = np.load(fn)
		except ValueError:
			continue
		k+=1
		label = int(ff[-2])
		y.append(label)
		data = normalize_data(data)
		data = rebin_arr(data, 64, 250)

		data_full.append(data)

	ndm, ntimes = data.shape

	data_full = np.concatenate(data_full, axis=0)
	data_full.shape = (k, -1)

	return data_full, np.array(y)

def combine_data_FT(fn):
	""" combine_data_FT('./single_pulse_ml/data/data_list')
	"""
	f = open(fn,'r')

	# data and its label class
	data_full, y = [], []

	for ff in f:
		line = ff.split(' ')

		fn, label = line[0], int(line[1])

		y.append(label)
		print(fn)
		tstamp = fn.split('+')[-2]
				
		#fdm = glob.glob('./*DM-T*%s*.npy' % tstamp)
		fn = './single_pulse_ml/data/test/' + fn
		data = read_pathfinder_npy(fn)
		data = normalize_data(data)
		data_full.append(data)
	
	nfreq, ntimes = data.shape[0], data.shape[-1]

	data_full = np.concatenate(data_full, axis=0)
	data_full.shape = (-1, nfreq*ntimes)

	return data_full, np.array(y)

def write_data(data, y, fname='out'):
	training_arr = np.concatenate((data, y[:, None]), axis=-1)

	np.save(fname, training_arr)


def read_data(fn):
	arr = np.load(fn)
	data, y = arr[:, :-1], arr[:, -1]

	return data, y

def read_pkl(fn):
	if fn[-4:]!='.pkl': fn+='.pkl'

	file = open(fn, 'rb')

	model = pickle.load(file)

	return model

def write_pkl(model, fn):
	if fn[-4:]!='.pkl': fn+='.pkl'
	
	file = open(fn, 'wb')
	pickle.dump(model, file)

	print("Wrote to pkl file: %s" % fn)

def get_labels():
	""" Cross reference DM-T files with Freq-T 
		files and create a training set in DM-T space. 
	"""

	fin = open('./single_pulse_ml/data/data_list','r')
	fout = open('./single_pulse_ml/data/data_list_DM','a')

	for ff in fin:
		x = ff.split(' ')
		n, c = x[0], int(x[1])
		try:
			t0 = n.split('+')[-2]
			float(t0)
		except ValueError:
			t0 = n.split('+')[-1].split('s')[0]

		newlist = glob.glob('./single_pulse_ml/data/DM*DM*%s*' % t0)

		if len(newlist) > 0:
			string = "%s %s\n" % (newlist[0].split('/')[-1], c)
			fout.write(string)

def create_training_set(FT=True, fout='./single_pulse_ml/data/data_freqtime_train'):
	if FT:
		data, y = combine_data_FT()
	else:
		data, y = combine_data_DT()

	write_data(data, y, fname=fout)







