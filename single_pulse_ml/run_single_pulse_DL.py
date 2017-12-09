""" Script to train and test or multiple deep 
    neural networks. 
"""

import numpy as np 
import time

import frb_keras_convnet 

FREQTIME=False   # train 2D frequency-time CNN
TIME1D=True      # train 1D pulse-profile CNN
DMTIME=False     # train 2D DM-time CNN
MULTIBEAM=True  # train feed-forward NN on simulated multibeam data

# Input hdf5 file. 
fn = "./data/_data_nt250_nf32_dm0_snrmax150.hdf5"
fn = "./data/_data_nt250_nf32_dm0_snrmax175.hdf5"
fn = "./data/_data_real_pf_pulses_.hdf5"
fn = "./data/_data_nt250_nf32_dm0_hybrid_pulse_simulation.hdf5"
fn = "./data/_data_nt250_nf32_dm0_snrmax150.hdf5"
fn = "./data/_data_nt250_nf32_dm0_snrmax60.hdf5"
fn = "./data/_data_nt250_nf32_dm0_snrmax75.hdf5"
#fn = './data/IAB_labeled.hdf5'

# Save tf model as .hdf5
save_model = True
fnout = "./model/keras_model_real_pulses"

NDM=300         # number of DMs in input array
NFREQ=32        # number of freq samples to use
NTIME=250       # number of time samples of input array
WIDTH=64        # width to use of arrays along time axis 
train_size=0.50 # fraction of dataset to train on

ftype = fn.split('.')[-1]

# Create empty lists for final merged model
model_list = []        
train_data_list = []
eval_data_list = []

# Configure the accuracy metric for evaluation
metrics = ["accuracy", "precision", "false_negatives", "recall"] 

if __name__=='__main__':
    # read in time-freq data, labels, dm-time data
    data_freq, y, data_dm = frb_keras_convnet.read_hdf5(fn)

    # low time index, high time index
    tl, th = NTIME//2-WIDTH//2, NTIME//2+WIDTH//2

    if data_freq.shape[-1] > (th-tl):
        data_freq = data_freq[..., tl:th]
        
    if data_dm.shape[-1] > (th-tl):
        data_dm = data_dm[:, :, tl:th]
    
    if data_dm.shape[-2] > 100:
        data_dm = data_dm[:, NDM//2-50:NDM//2+50]

    # tf/keras expects 4D tensors
    data_dm = data_dm[..., None]
    data_freq = data_freq[..., None]
    data_1d = data_freq.mean(1)

    # total number of triggers
    NTRIGGER = len(y)

    # fraction of true positives vs. total triggers
    TP_FRAC = np.float(y.sum()) / NTRIGGER

    # number of events on which to train
    NTRAIN = int(train_size * NTRIGGER)

    ind = np.arange(NTRIGGER)
    np.random.shuffle(ind)

    ind_train = ind[:NTRAIN]
    ind_eval = ind[NTRAIN:]

    # split up data into training and evaluation sets
    train_data_freq, eval_data_freq = data_freq[ind_train], data_freq[ind_eval]
    train_data_dm, eval_data_dm = data_dm[ind_train], data_dm[ind_eval]
    train_data_1d, eval_data_1d = data_1d[ind_train], data_1d[ind_eval]

    train_labels, eval_labels = y[ind_train], y[ind_eval]
    
    # Convert labels (integers) to binary class matrix
    train_labels = frb_keras_convnet.keras.utils.to_categorical(train_labels)
    eval_labels = frb_keras_convnet.keras.utils.to_categorical(eval_labels)

    if FREQTIME is True:
        print("Learning frequency-time array")
        # Build and train 2D CNN
        model_2d_freq_time, score_freq_time = frb_keras_convnet.construct_conv2d(
                        features_only=False, fit=True,
                        train_data=train_data_freq, eval_data=eval_data_freq, 
                        train_labels=train_labels, eval_labels=eval_labels,
                        epochs=5, nfilt1=32, nfilt2=64, 
                        nfreq=NFREQ, ntime=WIDTH)

        model_list.append(model_2d_freq_time)
        train_data_list.append(train_data_freq)
        eval_data_list.append(eval_data_freq)

        if save_model is True:
            fnout_freqtime = fnout+'freq_time.hdf5'
            model_2d_freq_time.save(fnout_freqtime)

    if DMTIME is True:
        print("Learning DM-time array")

        # Build and train 2D CNN
        model_2d_dm_time, score_dm_time = frb_keras_convnet.construct_conv2d(
                        features_only=False, fit=True,
                        train_data=train_data_dm, eval_data=eval_data_dm, 
                        train_labels=train_labels, eval_labels=eval_labels,
                        epochs=5, nfilt1=32, nfilt2=64, 
                        nfreq=NDM, ntime=WIDTH)
    
        model_list.append(model_2d_dm_time)
        train_data_list.append(train_data_dm)
        eval_data_list.append(eval_data_dm)

        if save_model is True:
            fnout_dmtime = fnout+'dm_time.hdf5'
            model_2d_dm_time.save(fnout_dmtime)

    if TIME1D is True:
        print("Learning pulse profile")

        # Build and train 1D CNN
        model_1d_time, score_1d_time = frb_keras_convnet.construct_conv1d(
                        features_only=False, fit=True,
                        train_data=train_data_1d, eval_data=eval_data_1d, 
                        train_labels=train_labels, eval_labels=eval_labels,
                        NTIME=64, nfilt1=64, nfilt2=128) 

        model_list.append(model_1d_time)
        train_data_list.append(train_data_1d)
        eval_data_list.append(eval_data_1d)

        if save_model is True:
            fnout_1dtime = fnout+'1d_time.hdf5'
            model_1d_time.save(fnout_1dtime)

    if MULTIBEAM is True:
        print("Learning multibeam data")

        # Right now just simulate multibeam, simulate S/N per beam.
        import simulate_multibeam as sm 

        nbeam = 32
        # Simulate a multibeam dataset
        data_mb, labels_mb = sm.make_dataset(ntrigger=NTRIGGER)
        data_mb_fp = data_mb[labels_mb[:,1]==0]
        data_mb_tp = data_mb[labels_mb[:,1]==1]

        train_data_mb = np.zeros([NTRAIN, nbeam])
        eval_data_mb = np.zeros([NTRIGGER-NTRAIN, nbeam])

        data_ = np.empty_like(data_mb)
        labels_ = np.empty_like(labels_mb)

        kk, ll = 0, 0
        for ii in range(NTRAIN):
            if train_labels[ii,1]==0:
                train_data_mb[ii] = data_mb_fp[kk]
                kk+=1
            elif train_labels[ii,1]==1:
                train_data_mb[ii] = data_mb_tp[ll]
                ll+=1

        for ii in range(NTRIGGER-NTRAIN):
            if eval_labels[ii,1]==0:
                eval_data_mb[ii] = data_mb_fp[kk]
                kk+=1
            elif eval_labels[ii,1]==1:
                eval_data_mb[ii] = data_mb_tp[ll]
                ll+=1

        model_mb, score_mb = frb_keras_convnet.construct_ff1d(
                             features_only=False, fit=True, 
                             train_data=train_data_mb, train_labels=train_labels,
                             eval_data=eval_data_mb, eval_labels=eval_labels,
                             nbeam=32, epochs=5,
                             nlayer1=32, nlayer2=32, batch_size=32)

        model_list.append(model_mb)
        train_data_list.append(train_data_mb)
        eval_data_list.append(eval_data_mb)

        if save_model is True:
            fnout_mb = fnout+'_mb.hdf5'
            model_mb.save(fnout_mb)

    if len(model_list)==1:
        score = model_list[0].evaluate(eval_data_list[0], eval_labels, batch_size=32)
        prob, predictions, mistakes = frb_keras_convnet.get_predictions(
                                model_list[0], eval_data_list[0], 
                                true_labels=eval_labels, epochs=10)
        print(mistakes)
        print("" % score)

    elif len(model_list)>1:
        print("\n=================================")
        print("    Merging & training %d models" % len(model_list))
        print("=================================\n")

        model, score = frb_keras_convnet.merge_models(
                                         model_list, train_data_list, 
                                         train_labels, eval_data_list, eval_labels)

        prob, predictions, mistakes = frb_keras_convnet.get_predictions(
                                model, eval_data_list, 
                                true_labels=eval_labels[:, 1])

        time.sleep(2)
        print('==========Results==========')
        try:
            print("\nFreq-time accuracy: %f" % score_freq_time[1])
        except:
            pass

        try:
            print("DM-time accuracy: %f" % score_dm_time[1])
        except:
            pass        

        try:
            print("Pulse-profile accuracy: %f" % score_1d_time[1])
        except:
            pass

        try:
            print("Multibeam accuracy: %f" % score_mb[1])
        except:
            pass

        print("\nMerged NN accuracy: %f" % score[1])
        print("\nIndex of mistakes: %s\n" % mistakes)
        frb_keras_convnet.print_metric(eval_labels[:, 1], predictions)






