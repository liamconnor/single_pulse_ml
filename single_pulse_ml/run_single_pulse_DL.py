""" Script to train and test or multiple deep 
    neural networks. Input models are expected to be 
    sequential keras model saved as an .hdf5 file.
    Input data are .hdf5 files with data sets
    'labels', 'data_freq_time', 'data_dm_time'. 
    They can be read by single_pulse_ml.reader.read_hdf5

    --How can this be made more portable? 
    --What data inputs should it be able to accept? 
    --Should it have a filterbank reader / dedisperser like 
    in triggers.py? maybe, not clear
    --Should I make a separate script for classification only? probably

"""
import os
import sys

import numpy as np 
import time
import h5py

from single_pulse_ml import plot_tools
from single_pulse_ml import reader
from single_pulse_ml import frbkeras

#try:
#    import matplotlib 
#    matplotlib.use('Agg')

#    import matplotlib.pyplot as plt
#    from matplotlib import gridspec
#    print("Worked")
#except:
#    "Didn't work"
#    pass

FREQTIME=True     # train 2D frequency-time CNN
TIME1D=False      # train 1D pulse-profile CNN
DMTIME=False      # train 2D DM-time CNN
MULTIBEAM=False   # train feed-forward NN on simulated multibeam data

# If True, the different nets will be clipped after 
# feature extraction layers and will not be compiled / fit
MERGE=False

CLASSIFY_ONLY=True
model_nm = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model", "keras_model_20000_arts")
prob_threshold = 0.1

# Input hdf5 file. 
#fn = './data/arts_b0329_only.hdf5'
#fn = '/data/03/Triggers/140514/oostrum/all_data.hdf5'
fn = sys.argv[1]

# Save tf model as .hdf5
save_model = True
fnout = "./model/keras_model_arts"

NDM=300          # number of DMs in input array
WIDTH=64         # width to use of arrays along time axis 
train_size=0.5   # fraction of dataset to train on

ftype = fn.split('.')[-1]

# Create empty lists for final merged model
model_list = []        
train_data_list = []
eval_data_list = []

# Configure the accuracy metric for evaluation
metrics = ["accuracy", "precision", "false_negatives", "recall"] 

if __name__=='__main__':
    # read in time-freq data, labels, dm-time data
    data_freq, y, data_dm, params = reader.read_hdf5(fn, read_params=True)

    #data_freq = data_freq[1::2]
    #y = y[1::2]

    print("Using %s" % fn)

    NFREQ = data_freq.shape[1]
    NTIME = data_freq.shape[2]

    # low time index, high time index
    tl, th = NTIME//2-WIDTH//2, NTIME//2+WIDTH//2

    if data_freq.shape[-1] > (th-tl):
        data_freq = data_freq[..., tl:th]

    dshape = data_freq.shape

    # normalize data
    data_freq = data_freq.reshape(len(data_freq), -1)
    data_freq -= np.median(data_freq, axis=-1)[:, None]
    data_freq /= np.std(data_freq, axis=-1)[:, None]

    # zero out nans
    data_freq[data_freq!=data_freq] = 0.0
    data_freq = data_freq.reshape(dshape)

    if DMTIME is True:
        if data_dm.shape[-1] > (th-tl):
            data_dm = data_dm[:, :, tl:th]
    
        if data_dm.shape[-2] > 100:
            data_dm = data_dm[:, NDM//2-50:NDM//2+50]
    
        # tf/keras expects 4D tensors
        data_dm = data_dm[..., None]

    if FREQTIME is True:
        # tf/keras expects 4D tensors
        data_freq = data_freq[..., None]
        data_1d = data_freq.mean(1)

    if CLASSIFY_ONLY is False:
        # total number of triggers
        NTRIGGER = len(y)

        # fraction of true positives vs. total triggers
        TP_FRAC = np.float(y.sum()) / NTRIGGER

        # number of events on which to train
        NTRAIN = int(train_size * NTRIGGER)

        ind = np.arange(NTRIGGER)
        np.random.shuffle(ind)

        # hack
        #ind = list(np.arange(0, NTRIGGER, 2)) + list(np.arange(1, NTRIGGER, 2))

        ind_train = ind[:NTRAIN]
        ind_eval = ind[NTRAIN:]

        train_labels, eval_labels = y[ind_train], y[ind_eval]
        
        # Convert labels (integers) to binary class matrix
        train_labels = frbkeras.keras.utils.to_categorical(train_labels)
        eval_labels = frbkeras.keras.utils.to_categorical(eval_labels)


    if FREQTIME is True:

        if CLASSIFY_ONLY is True:
            print("Classifying freq-time data")
            model_freq_time_nm = model_nm + 'freq_time.hdf5'
            eval_data_list.append(data_freq)

            model_freq_time = frbkeras.load_model(model_freq_time_nm)
            y_pred_prob = model_freq_time.predict(data_freq)
            y_pred_prob = y_pred_prob[:,1]
            y_pred_freq_time = np.round(y_pred_prob)

            ind_frb = np.where(y_pred_prob>prob_threshold)[0]

            frbkeras.print_metric(y, y_pred_freq_time)

            print("\n%d out of %d events with probability > %.2f:\n %s" % 
                    (len(ind_frb), len(y_pred_prob), 
                        prob_threshold, ind_frb))

            low_to_high_ind = np.argsort(y_pred_prob)
            fnout_ranked = fn.rstrip('.hdf5') + 'freq_time_candidates.hdf5'

            g = h5py.File(fnout_ranked, 'w')
            g.create_dataset('data_frb_candidate', data=data_freq[ind_frb])
            g.create_dataset('frb_index', data=ind_frb)
            g.create_dataset('probability', data=y_pred_prob[ind_frb])
            g.create_dataset('params', data=params[ind_frb])
            g.close()

            #plot_tools.plot_ranked_trigger(data_freq[..., 0], 
            #            y_pred_prob[:, None], h=5, w=5, ascending=False, 
            #            outname='out')

            print("\nSaved them and all probabilities to: \n%s" % fnout_ranked)
        else:
            print("Learning frequency-time array")

            # split up data into training and evaluation sets
            train_data_freq, eval_data_freq = data_freq[ind_train], data_freq[ind_eval]

            # Build and train 2D CNN
            model_2d_freq_time, score_freq_time = frbkeras.construct_conv2d(
                            features_only=MERGE, fit=True,
                            train_data=train_data_freq, eval_data=eval_data_freq, 
                            train_labels=train_labels, eval_labels=eval_labels,
                            epochs=5, nfilt1=32, nfilt2=64, 
                            nfreq=NFREQ, ntime=WIDTH)

            model_list.append(model_2d_freq_time)
            train_data_list.append(train_data_freq)
            eval_data_list.append(eval_data_freq)

            if save_model is True:
                fnout_freqtime = fnout + 'freq_time.hdf5'
                model_2d_freq_time.save(fnout_freqtime)
                print("Saving freq-time model to: %s" % fnout_freqtime)

    if DMTIME is True:
        if CLASSIFY_ONLY is True:
            print("Classifying dm-time data")

            model_dm_time_nm = model_nm + 'dm_time.hdf5'
            eval_data_list.append(data_dm)

            model_dm_time = frbkeras.load_model(model_dm_time_nm)
            y_pred_prob = model_dm_time.predict(data_dm)
            y_pred_dm_time = np.round(y_pred_prob[:,1])

            print("\nMistakes: %s" % np.where(y_pred_dm_time!=y)[0])

            frbkeras.print_metric(y, y_pred_dm_time)
            print("") 
        else:
            print("Learning DM-time array")
            # split up data into training and evaluation sets
            train_data_dm, eval_data_dm = data_dm[ind_train], data_dm[ind_eval]

            # split up data into training and evaluation sets
            train_data_dm, eval_data_dm = data_dm[ind_train], data_dm[ind_eval]

            # Build and train 2D CNN
            model_2d_dm_time, score_dm_time = frbkeras.construct_conv2d(
                            features_only=MERGE, fit=True,
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
                print("Saving dm-time model to: %s" % fnout_dmtime)

    if TIME1D is True:

        if CLASSIFY_ONLY is True:
            print("Classifying pulse profile")

            model_time_nm = model_nm + '1d_time.hdf5'
            eval_data_list.append(data_1d)

            model_1d_time = frbkeras.load_model(model_time_nm)
            y_pred_prob = model_1d_time.predict(data_1d)
            y_pred_time = np.round(y_pred_prob[:,1])

            print("\nMistakes: %s" % np.where(y_pred_time!=y)[0])

            frbkeras.print_metric(y, y_pred_time)
            print("") 
        else:
            print("Learning pulse profile")            
            # split up data into training and evaluation sets
            train_data_1d, eval_data_1d = data_1d[ind_train], data_1d[ind_eval]

            # Build and train 1D CNN
            model_1d_time, score_1d_time = frbkeras.construct_conv1d(
                            features_only=MERGE, fit=True,
                            train_data=train_data_1d, eval_data=eval_data_1d, 
                            train_labels=train_labels, eval_labels=eval_labels,
                            nfilt1=64, nfilt2=128) 

            model_list.append(model_1d_time)
            train_data_list.append(train_data_1d)
            eval_data_list.append(eval_data_1d)

            if save_model is True:
                fnout_1dtime = fnout+'1d_time.hdf5'
                model_1d_time.save(fnout_1dtime)
                print("Saving 1d-time model to: %s" % fnout_1dtime)

    if MULTIBEAM is True:
        print("Learning multibeam data")

        # Right now just simulate multibeam, simulate S/N per beam.
        import simulate_multibeam as sm 

        nbeam = 32
        # Simulate a multibeam dataset
        data_mb, labels_mb = sm.make_multibeam_data(ntrigger=NTRIGGER)
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

        model_mb, score_mb = frbkeras.construct_ff1d(
                                    features_only=MERGE, fit=True, 
                                    train_data=train_data_mb, 
                                    train_labels=train_labels,
                                    eval_data=eval_data_mb, 
                                    eval_labels=eval_labels,
                                    nbeam=nbeam, epochs=5,
                                    nlayer1=32, nlayer2=32, batch_size=32)

        model_list.append(model_mb)
        train_data_list.append(train_data_mb)
        eval_data_list.append(eval_data_mb)

        if save_model is True:
            fnout_mb = fnout+'_mb.hdf5'
            model_mb.save(fnout_mb)

    if len(model_list)==1:
        score = model_list[0].evaluate(eval_data_list[0], eval_labels, batch_size=32)
        prob, predictions, mistakes = frbkeras.get_predictions(
                                model_list[0], eval_data_list[0], 
                                true_labels=eval_labels)
        print(mistakes)
        print("" % score)

    elif MERGE is True:

        if CLASSIFY_ONLY is True:
            print("Classifying merged model")
            model_merged_nm = model_nm + '_merged.hdf5'

            model_merged = frbkeras.load_model(model_merged_nm)
            y_pred_prob = model_merged.predict(data_list)
            y_pred = np.round(y_pred_prob[:,1])

            print("Mistakes: %s" % np.where(y_pred!=y)[0])
            frbkeras.print_metric(y, y_pred)
            print("") 
        else:

            print("\n=================================")
            print("    Merging & training %d models" % len(model_list))
            print("=================================\n")

            model, score = frbkeras.merge_models(
                                             model_list, train_data_list, 
                                             train_labels, eval_data_list, eval_labels,
                                             epochs=5)

            prob, predictions, mistakes = frbkeras.get_predictions(
                                    model, eval_data_list, 
                                    true_labels=eval_labels[:, 1])


            if save_model is True:
                fnout_merged = fnout+'_merged.hdf5'
                model.save(fnout_merged)

            print("\nMerged NN accuracy: %f" % score[1])
            print("\nIndex of mistakes: %s\n" % mistakes)
            frbkeras.print_metric(eval_labels[:, 1], predictions)

    if CLASSIFY_ONLY is False:
        print('\n==========Results==========')
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


