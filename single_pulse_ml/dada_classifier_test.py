#!/usr/bin/env python3
import numpy as np
from psrdada import Reader
import time
import matplotlib.pylab as plt
import logging

import realtime_tools
import frbkeras

fn_model = 'model/20190125-17114-freqtimefreq_time_model.hdf5'
triggermode = True 
nfreq_plot = 32
ntime_plot = 64
dt = 8.192e-5

RtProc = realtime_tools.RealtimeProc()

model = frbkeras.load_model(fn_model)

# For some reason, the model's first prediction takes a long time. 
# pre-empt this by classifying an array of zeros before looking 
# at real data
model.predict(np.zeros([1, nfreq_plot, ntime_plot, 1]))

reader = Reader()

def dada_proc_trigger(reader):
    # Connect to a running ringbuffer with key=1200
    reader.connect(0x1200)
    counter = -1

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
            dshape = (H.ntab, H.nchan, H.ntime_batch)
            tab = H.beamno
            snr = H.snr

        data = np.reshape(data, dshape)

        logging.info("Dedispersing to dm=%0.1f at t=%0.1fsec with width=%.1f S/N=%.1f" %
                         (dm_, t0, width, snr))
        print(counter, dm, width, tab, H.astropy_page_time)
        dm = 0.
        width = 10

        #data[:, :, int(H.ntime_batch/2):10+int(H.ntime_batch/2)] += 5

        if len(data)==0:
            continue

        # This method will rfi clean, dedisperse, and downsample data.
        data_classify, data_dmtime = RtProc.proc_all(data[tab], dm, nfreq_plot=nfreq_plot, 
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

dada_proc_trigger(reader)










