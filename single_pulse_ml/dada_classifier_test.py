#!/usr/bin/env python3
import numpy as np
from psrdada import Reader
import time
import matplotlib.pylab as plt

import realtime_tools
import frbkeras

fn_model = 'model/20190125-17114-freqtimefreq_time_model.hdf5'
triggermode = True 
nfreq_plot = 32
ntime_plot = 64
ntab = 12
dt = 8.192e-5
counter = -1

RtProc = realtime_tools.RealtimeProc()

model = frbkeras.load_model(fn_model)

# For some reason, the model's first prediction takes a long time. 
# pre-empt this by classifying an array of zeros before looking 
# at real data
model.predict(np.zeros([1, nfreq_plot, ntime_plot, 1]))

while True:
    while not reader.isEndOfData:
        # read the page as numpy array
        page = reader.getNextPage()

        data = np.asarray(page)
        print(np.sum(data))

        reader.markCleared()
        # Create a reader instace
        reader = Reader()

    continue

    # Connect to a running ringbuffer with key=1200
    reader.connect(0x1200)

    for page in reader:
        print(len(page))

    reader.disconnect()
    print('disconned')
    continue
    for page in reader:
        t0 = time.time()
        counter += 1
        data = np.array(page)

        if counter==0:
            header = reader.getHeader()
            H = realtime_tools.DadaHeader(header, trigger=triggermode)
            dm = H.dm
            width = np.int(H.width)
            t_batch = H.ntime_batch*H.dt
            dshape = (ntab, H.nchan, H.ntime_batch)
            tab = H.beamno
        
        data = np.reshape(data, dshape)
        print(counter, dm, width, tab, H.astropy_page_time)
        dm = 0.
        width = 10

        data[:, :, int(H.ntime_batch/2):10+int(H.ntime_batch/2)] += 5

        if len(data)==0:
            continue

        # This method will rfi clean, dedisperse, and downsample data.
        data_classify, data_dmtime = RtProc.proc_all(data[:], dm, nfreq_plot=nfreq_plot, 
                                                     ntime_plot=ntime_plot, 
                                                     invert_spectrum=True, 
                                                     downsample=width, dmtransform=True)

        prob = model.predict(data_classify[..., None])
        indpmax = np.argmax(prob[:, 1])

        print('t PROC: %f' % (time.time()-t0))
        if prob[indpmax,1]>0.25:
            fig = plt.figure()
            plt.imshow(data_classify[indpmax], aspect='auto')
            plt.title(str(prob.max()))
            plt.show()
        else:
            print('Nothing good')

    reader.disconnect()
