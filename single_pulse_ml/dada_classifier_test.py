#!/usr/bin/env python3
import numpy as np
from psrdada import Reader

import matplotlib.pylab as plt

import tools3
import frbkeras

fn_model = 'model/20190125-17114-freqtimefreq_time_model.hdf5'
model = frbkeras.load_model(fn_model)

# Create a reader instace
#reader = Reader()

# Connect to a running ringbuffer with key 'dada'
#reader.connect(0xdada)
#reader.connect(0x1210)

ntab = 3
nfreq = 1536
ntime_batch = 12500
dshape = (ntab, nfreq, ntime_batch)

counter = -1

def dedisperse(data, dm, dt=8.192e-5, freq=(1550, 1250), freq_ref=None):
    data = data.copy()
    
    nfreq, ntime = data.shape[0], data.shape[1]

    freqs = np.linspace(freq[0], freq[-1], nfreq)

    if freq_ref is None:
        freq_ref = freqs.max()

    tdelay = 4.148e3*dm*(freqs**-2 - freq_ref**-2)
    ntime = len(data[0])

    maxind_arr = []

    for ii, f in enumerate(freqs):
        data[ii] = np.roll(data[ii], -np.int(tdelay[ii]/dt))

    return data

def preprocess(data, invert_spectrum=False):
    if len(data.shape)==2:
        data = data[None]

    if invert_spectrum:
        data = data[:, ::-1]

    ntab = data.shape[0]

    for tab in range(ntab):
        data_tab = data[tab]
        data_tab = tools3.cleandata(data_tab, threshold=3.0)
        data[tab] = data_tab

    if data.shape[0]==1:
        data = data[0]

    return data 

def dedisperse_tabs(data, dm, dt=8.192e-5, freq=(1550, 1250), freq_ref=None):

    if len(data.shape)==2:
        data = data[None]

    ntab = data.shape[0]

    for tab in range(ntab):
        data_tab = data[tab]
        data_tab = dedisperse(data_tab, dm, dt=dt, freq=freq, freq_ref=freq_ref)
        data[tab] = data_tab

    if data.shape[0]==1:
        data = data[0]

    return data

def postprocess(data, nfreq_plot=32, ntime_plot=64, downsample=1):
    if len(data.shape)==2:
        data = data[None]

    nfreq = data.shape[1]
    ntime = data.shape[-1]
    ntab = data.shape[0]
    data_classify = np.empty([ntab, nfreq_plot, ntime_plot])

    for tab in range(ntab):
        data_tab = data[tab]
        print(tab, data_tab.shape, data_classify.shape)
        data_tab = data_tab.reshape(nfreq_plot, nfreq//nfreq_plot, -1).mean(1)
        data_tab = data_tab[:, :ntime//downsample*downsample]
        data_tab = data_tab.reshape(-1, ntime//downsample, downsample).mean(-1)

        maxind = np.argmax(data_tab.mean(0))

        data_tab -= np.median(data_tab)
        data_tab /= np.std(data_tab)
        data_tab[data_tab!=data_tab] = 0.

        data_classify[tab] = data_tab[:, maxind-ntime_plot//2:maxind+ntime_plot//2]

    return data_classify

reader = []

for ii in range(5):
    print(ii)
    data = np.random.normal(0,1,1536*ntab*ntime_batch)
    reader.append(data)

for page in reader:
    print(counter)
    counter += 1
    
    # read the page as numpy array
    data = (np.asarray(page)).copy()

#    header = reader.getHeader()

    if len(data)>0:
        print(data.shape, counter)
    else:
        continue

    data = np.reshape(data, dshape)
#    data = data[:, ::-1]

    data = preprocess(data, invert_spectrum=True)
    data = dedisperse_tabs(data, 56.0)
    data_classify = postprocess(data, nfreq_plot=32, ntime_plot=64, downsample=1)

    prob = model.predict(data_classify[..., None])
    indpmax = np.argmax(prob[:, 1])

    if prob[indpmax,1]>=0.0:
        fig = plt.figure()
        plt.imshow(data_classify[indpmax], aspect='auto', vmax=3, vmin=-2.)
        plt.show()
    else:
        print('Nothing good')

#    data, maxind_arr = dedisperse(data, dm=56.8)
#    data = data[..., :ntime_batch//64*64]
#    data = data.reshape(ntab, 32, nfreq//32, 64, ntime_batch//64)
#    data = data.mean(2).mean(-1)
#     data = data.reshape(ntab, 32, nfreq//32, -1).mean(2)
#     for tab in range(ntab):
#         data_tab = data[tab]
#         data_tab = tools3.cleandata(data_tab, threshold=3.0)
#         data_tab = tools3.dedisperse(data_tab, dm, dt=8.192e-5, freq=(1550, 1250), freq_ref=None)
#         data_tab = data_tab.reshape(32, nfreq//32, -1).mean(1)
#         data_tab = data_tab[:, :ntime_batch//downsample*downsample]
#         data_tab = data_tab.reshape(-1, ntime_batch//downsample, downsample).mean(-1)

#         maxind = np.argmax(data_tab.mean(0))

#         data_tab -= np.median(data_tab)
#         data_tab /= np.std(data_tab)
#         data_tab[data_tab!=data_tab] = 0.
#         data_classify[tab] = data_tab[:, maxind-32:maxind+32]
        
#     prob = model.predict(data_classify[..., None])
#     indpmax = np.argmax(prob[:, 1])

#     if prob[indpmax,1]>=0.0:
#         fig = plt.figure()
#         plt.imshow(data_classify[indpmax], aspect='auto', vmax=3, vmin=-2.)
#         plt.show()
#     else:
#         print('Nothing good')

# reader.disconnect()
