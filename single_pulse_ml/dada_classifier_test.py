#!/usr/bin/env python3
import numpy as np
from psrdada import Reader
import time
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

def do_all_rename(data, dm, nfreq_plot=32, ntime_plot=64):
    data = RtProc.preprocess(data, invert_spectrum=True)
    data = RtProc.dedisperse_tabs(data, dm)
    data_classify = RtProc.postprocess(data, nfreq_plot=nfreq_plot, ntime_plot=ntime_plot, downsample=1)
    prob = model.predict(data_classify[..., None])
    return data, prob

freq_high=1549.6
freq_low=1249.6
ntab = 12
nfreq = 1536
ntime_batch = 12500
dshape = (ntab, nfreq, ntime_batch)
dt = 8.192e-5
t_batch = ntime_batch*dt

RtProc = tools3.RealtimeProc()

counter = -1
reader = []

# Generate fake pages of data
for ii in range(3):
    print(ii)
    data = np.random.normal(0,1,1536*ntab*ntime_batch)
    reader.append(data)

dm = 5000.
to_list = [1., 1.25, 1.5]
dm = [100,500,250.]

# Leaving this as is. Need to think of how I will 
# actually read the data as it comes in. 
# Ask Leon again how the trigger headers actually work.
# Need to figure out how many triggers per minute the code 
# can keep up with. Also need to update the dedisperser for edge 
# effect. The frequency roll should be more like presto. .


for page in reader:
    
    counter += 1
    times = np.linspace(0, counter*t_batch, n_batch)
    
    for ii, to in enumerate(to_list):
        t_disp = np.abs(4148*dm[ii]*(freq_high**-2 - freq_low**-2))
        if not to>times[0] and (to+t_disp)<times[-1]:
            continue

    t_disp = np.abs(4148*dm*(freq_high**-2 - freq_low**-2))
    n_page_event = int(t_disp/t_batch)

    if event_counter==0:
        t_disp = np.abs(4148*dm*(freq_high**-2 - freq_low**-2))
        n_page_event = int(t_disp/t_batch)
        t_batch = n_batch*dt
    else:
        t_batch += n_batch*dt

    data = np.reshape(data, dshape)
    # read the page as numpy array
#    data = (np.asarray(page)).copy()
    if t_disp > t_batch:
        print(event_counter, t_batch)
        data_old.append(page)
        event_counter += 1
        continue
    else:
        event_counter = 0
        data = page
        data_old = []

#    header = reader.getHeader()

    if len(data)>0:
        print(data.shape, counter)
    else:
        continue

    data, prob = do_all_rename(data, dm, nfreq_plot=32, ntime_plot=64)

    prob = model.predict(data_classify[..., None])
    indpmax = np.argmax(prob[:, 1])

    if prob[indpmax,1]>1.0:
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
