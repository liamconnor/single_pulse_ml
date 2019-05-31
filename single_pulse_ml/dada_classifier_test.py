#!/usr/bin/env python3
import os

import numpy as np
from psrdada import Reader
import time
import matplotlib.pylab as plt
import logging

import realtime_tools
import frbkeras
import simulate_frb2 # test hack

os.system('./disk_to_buffer_tests.sh &')

logfn = time.strftime("%Y%m%d-%H%M") + '.log'
logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO, filename=logfn)

fn_model_freqtime = 'model/20190125-17114-freqtimefreq_time_model.hdf5'
fn_model_dmtime = 'model/heimdall_dm_time.hdf5'
triggermode = True 
nfreq_plot = 32
ntime_plot = 64
ndm_plot = 64
dt = 8.192e-5

RtProc = realtime_tools.RealtimeProc()

model_freqtime = frbkeras.load_model(fn_model_freqtime)
model_dmtime = frbkeras.load_model(fn_model_dmtime)

# For some reason, the model's first prediction takes a long time. 
# pre-empt this by classifying an array of zeros before looking 
# at real data
model_freqtime.predict(np.zeros([1, nfreq_plot, ntime_plot, 1]))
model_dmtime.predict(np.zeros([1, ndm_plot, ntime_plot, 1]))

reader = Reader()

def dada_proc_trigger(reader, nbeam=12):
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
            dshape = (nbeam, H.nchan, H.ntime_batch)
            tab = H.beamno
            snr = H.snr

        data = np.reshape(data, dshape)
        print(H.freq_high)
        A, p = simulate_frb2.gen_simulated_frb(fluence=1000, 
                                               dm=dm, width=0.001, 
                                               background_noise=data[tab].astype(float),
                                               NTIME=12500, 
                                               NFREQ=1536, freq=(1550., 1250))
        A[A>255] = 255
        A[A<0] = 0

        data = A.astype(data[-1].dtype)

        logging.info("Received dm=%0.1f at t=%0.1fsec with width=%.1f S/N=%.1f" %
                         (dm, t0, width, snr))

#        dm = 0
#        width = 10
#        data[:, :, int(H.ntime_batch/2):10+int(H.ntime_batch/2)] += 100

        if len(data)==0:
            continue

        fig = plt.figure()
        plt.imshow(data, aspect='auto')
        plt.show()
        # This method will rfi clean, dedisperse, and downsample data.
        data_classify, data_dmtime = RtProc.proc_all(data, dm, 
                                                     nfreq_plot=nfreq_plot, 
                                                     ntime_plot=ntime_plot, 
                                                     invert_spectrum=True, 
                                                     downsample=width, dmtransform=True)
        fig = plt.figure()
        plt.imshow(data_classify[0], aspect='auto')
        plt.show()

        prob_freqtime = model_freqtime.predict(data_classify[..., None])
        indpmax_freqtime = np.argmax(prob_freqtime[:, 1])

        prob_dmtime = model_dmtime.predict(data_dmtime[..., None])
        indpmax_dmtime = np.argmax(prob_dmtime[:, 1])

        logging.info("page %d proc time %0.2f" % (counter, time.time()-t0))

        if 1>0:
            fig, axes = plt.subplots(2, 1)
            axes[0].imshow(data_dmtime[indpmax_dmtime], aspect='auto')
#            axes[1].imshow(data_classify[indpmax_freqtime], aspect='auto')
            axes[1].imshow(data_classify[0], aspect='auto')

            axes[0].set_title(prob_dmtime[indpmax_dmtime, 1])
            axes[1].set_title(prob_freqtime[indpmax_freqtime, 1])

            axes[0].set_ylabel('DM', fontsize=18)
            axes[1].set_ylabel('Frequency', fontsize=18)

            plt.show()
        else:
            logging.info("Nothing")

    reader.disconnect()

dada_proc_trigger(reader)










