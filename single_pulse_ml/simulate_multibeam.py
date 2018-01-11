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

import frbkeras

def gauss(x, xo, sig):
    return np.exp(-(x-xo)**2/sig**2)

def generate_multibeam(nbeam=32, rows=8, cols=4, width=27, nside=1000):
    """ width in arcminutes
    """
    # convert arcminutes to degrees
    width /= 60. 

    # theta in degrees
    theta = np.linspace(-1, 1, 100)

    # compute 1D gaussian beam
    beam_theta = gauss(theta, 0, width)

    # compute 1D beam outer product with itself for 2D
    beam_2d = beam_theta[None]*beam_theta[:, None]

    # create nbeam arrays 
    beam_arr = np.zeros([nside, nside, nbeam])

    # Make each beam 
    kk=0
    for ii in range(rows):
        for jj in range(cols):
            # get x,y coordinates of each beam center
            xx, yy = 500-4*50+ii*50, 500-2*50+jj*50
            beam_arr[xx:xx+100, yy:yy+100, kk] += beam_2d
            kk+=1

    return beam_arr

def test_merge_model(n=32, m=64, ntrigger=10000):
    data = np.random.normal(0, 1, n*m*ntrigger).reshape(ntrigger, n, m)
    data[ntrigger//2:, :, m//2-2:m//2+1] += 0.25
    data /= np.std(data.reshape(-1, n*m), -1)[:, None, None]
    data -= np.median(data, 2)[:, :, None]

    # set RFI labels to 0, FRBs to 1
    labels = np.zeros([ntrigger])
    labels[ntrigger//2:] = 1 

    # convert to categorical array with shape (-1, 2)
    labels = labels.astype(int)
    labels = keras.utils.to_categorical(labels)

    data = data[..., None]

    model_2d_freq_time, score_freq_time = frbkeras.construct_conv2d(
                        features_only=False, fit=True,
                        train_data=data[::2], eval_data=data[1::2], 
                        train_labels=labels[::2], eval_labels=labels[1::2],
                        epochs=5, nfilt1=32, nfilt2=64, 
                        nfreq=n, ntime=m)
    print(score_freq_time)

    train_data_mb, train_labels, eval_data_mb, eval_labels, model_mb = run_model(ntrigger)

    model_list = [model_mb, model_2d_freq_time]
    train_data_list = [train_data_mb, data[::2]]
    eval_data_list = [eval_data_mb, data[1::2]]

    model, score = frbkeras.merge_models(model_list, train_data_list, 
                                         train_labels, eval_data_list, eval_labels,
                                         epoch=5)

    print(score)

    return data, labels, train_data_mb, train_labels, model

def make_multibeam_data(ntrigger=2304, tp_frac=0.5):
    A = generate_multibeam()
    # Take a euclidean flux distribution
    sn = np.random.uniform(1, 1000, 100*ntrigger)**-(2/3.) 
    sn /= np.median(sn)
    sn *= 15
    #sn[sn > 150] = 150

    det_ = []
    sn_ = []
    multis = 0

    # drop FRBs at random locations with random flux
    for ii, ss in enumerate(sn):
        xi = np.random.uniform(400, 650)
        yi = np.random.uniform(300, 750)
        abeams = A[int(xi), int(yi)] * ss
        beamdet = np.where(abeams>=6)[0]
        if len(beamdet)>0:
            det_.append(beamdet)
            sn_.append(abeams[beamdet])
            if len(beamdet)>1:
                multis += 1

    ntrigger = min(2*len(det_), ntrigger)
    nbeam = 32 # number of beams
    data = np.zeros([nbeam*ntrigger]).reshape(-1, nbeam)
    N_FP = int((1-tp_frac)*ntrigger)
    N_TP = int(tp_frac*ntrigger)

    for ii in range(N_FP):
#        nbeam_ii = int(np.random.uniform(1, 32))

        # Generate number of beams RFI shows up in
        nbeam_ii = min(nbeam, int(np.random.lognormal(1.25, 0.8))) 

        ind = set(np.random.uniform(1, 32, nbeam_ii).astype(int).astype(list))
        data[ii][list(ind)] = np.random.normal(20, 5, len(ind))

    for ii in range(N_TP):
#       beam = int(np.random.uniform(1, 32))
        data[N_FP+ii][det_[ii]] = sn_[ii]#np.random.normal(20, 5, 1)

    # set RFI labels to 0, FRBs to 1
    labels = np.zeros([ntrigger])
    labels[N_FP:] = 1 

    # convert to categorical array with shape (-1, 2)
    labels = labels.astype(int)
    labels = keras.utils.to_categorical(labels)

    # Print to see if fraction of multibeam detections is expected
    print(np.float(multis) / len(det_))

    return data, labels

def run_model(n):
    import frbkeras

    data_mb, labels = make_multibeam_data(ntrigger=n, tp_frac=0.5)
    train_data_mb = data_mb[::2]
    train_labels = labels[::2]
    eval_data_mb = data_mb[1::2]
    eval_labels = labels[1::2]

    model_mb, score_mb = frbkeras.construct_ff1d(
                                features_only=False, fit=True, 
                                train_data=train_data_mb, 
                                train_labels=train_labels,
                                eval_data=eval_data_mb, 
                                eval_labels=eval_labels,
                                nbeam=32, epochs=5,
                                nlayer1=32, nlayer2=32, 
                                batch_size=32)

    if len(score_mb)>1:
        prob, predictions, mistakes = frbkeras.get_predictions(
                                model_mb, eval_data_mb, 
                                true_labels=eval_labels)
    print(score_mb)

    return train_data_mb, train_labels, eval_data_mb, eval_labels, model_mb

