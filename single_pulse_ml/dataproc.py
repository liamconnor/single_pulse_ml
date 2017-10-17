""" Tools for preprocessing data
"""


import numpy as np

def normalize_data(data):
	""" Normalize data to zero-median and 
	unit standard deviation

	Parameters
	----------
	data : np.array
		(nfreq, ntimes)
	"""
	# subtract each channel's median
	data -= np.median(data, axis=-1)[:, None]
	# demand unit variance
#	data /= np.std(data, axis=-1)[:, None] 
# 	Try dividing by global variance.
	data /= np.std(data)
	# Replace nans with zero
	data[data!=data] = 0.

	return data


def dedisp(data, dm, freq=np.linspace(800, 400, 1024), dt=512*2.56e-6):
	""" Dedisperse data by shifting freq bins

	Parameters
	----------
	data : np.array
		(nfreq, ntimes)
	dm   : np.float
		dispersion measure in pc cm**-3
	freq : np.array
		(nfreq) vector in MHz
	dt   : np.float
		time resolution of data in seconds
	"""
	dm_del = 4.148808e3 * dm * (freq**(-2) - 600.0**(-2))
	data_out = np.zeros_like(data)

	for ii, ff in enumerate(freq):
		dmd = int(round(dm_del[ii] / dt))
		data_out[ii] = np.roll(data[ii], -dmd, axis=-1)

	return data_out

def dm_delays(dm, freq, f_ref):
	""" Calculate dispersion delays in seconds

	Parameters
	----------
	dm   : np.float
		dispersion measure in pc cm**-3
	freq : np.array
		(nfreq) vector in MHz
	f_ref: np.float
		reference frequency in MHz
	"""
	return 4.148808e3 * dm * (freq**(-2) - f_ref**(-2))


def straighten_arr(data):
	""" Step through each freq, find DM shift
	that gives largest S/N, realign bins

	Parameters
	----------
	data : np.array
		(nfreq, ntimes)
	"""

	sn = []

	dms = np.linspace(-5, 5, 100)

	for dm in dms:
		d_ = dedisp(data.copy(), dm, freq=linspace(800,400,16))
		sn.append(d_.mean(0).max() / np.std(d_.mean(0)))

	d_ = dedisp(data, dms[np.argmax(sn)], freq=linspace(800,400,16))

	return d_

def run_straightening(fn):
	""" Take filename, read in data, shift 
	to remove any excess dm-delay. 

	Parameters
	----------
	fn : str 
		filename of numpy array
	"""
	f = np.load(fn)

	y = f[:, -1]

	d = f[y==1, :-1].copy()

	for ii in range(len(d)):
		dd_ = d[ii].reshape(-1, 250)
		d[ii] = (straighten_arr(dd_)).reshape(-1)
		
	f[y==1, :-1] = d

	for jj in range(len(f)):
		dd_ = f[jj, :-1].reshape(-1, 250)
		dd_ = reader.normalize_data(dd_)
		f[jj, :-1] = dd_.flatten()

	return f