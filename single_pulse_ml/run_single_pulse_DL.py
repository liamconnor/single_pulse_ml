import numpy as np 

import frb_keras_convnet 

FREQTIME=False
DMTIME=True
TIME1D=True

fn = "./data/_data_nt250_nf32_dm0_snrmax150.hdf5"

NDM=150
NFREQ=32
NTIME=250
WIDTH=64
train_size=0.75
ftype = fn.split('.')[-1]

model_list = []
train_data_list = []
eval_data_list = []

# Configure the accuracy metric for evaluation
metrics = ["accuracy", "precision", "false_negatives", "recall"] 

if __name__=='__main__':
    data_freq, y, data_dm = frb_keras_convnet.read_hdf5(fn)
    
    tl, th = NTIME//2-WIDTH//2, NTIME//2+WIDTH//2

    data_freq = data_freq[..., tl:th]
    data_dm = data_dm[..., tl:th]
    
    # tf expects 4D tensors
    data_dm = data_dm[..., None]
    data_freq = data_freq[..., None]
    data_1d = data_freq.mean(1)

    NTRIGGER = len(y)
    NTRAIN = int(train_size * NTRIGGER)

    ind = np.arange(NTRIGGER)
    np.random.shuffle(ind)

    ind_train = ind[:NTRAIN]
    ind_eval = ind[NTRAIN:]

    train_data_dm, eval_data_dm = data_dm[ind_train], data_dm[ind_eval]
    train_data_freq, eval_data_freq = data_freq[ind_train], data_freq[ind_eval]
    train_data_1d, eval_data_1d = data_1d[ind_train], data_1d[ind_eval]

    train_labels, eval_labels = y[ind_train], y[ind_eval]
    
    train_labels = frb_keras_convnet.keras.utils.to_categorical(train_labels)
    eval_labels = frb_keras_convnet.keras.utils.to_categorical(eval_labels)

    if FREQTIME is True:

        model_2d_freq_time, score = frb_keras_convnet.construct_conv2d(
                        features_only=False, fit=True,
                        train_data=train_data_freq, eval_data=eval_data_freq, 
                        train_labels=train_labels, eval_labels=eval_labels,
                        epochs=5, nfilt1=32, nfilt2=64, 
                        nfreq=NFREQ, ntime=WIDTH)

        model_list.append(model_2d_freq_time)
        train_data_list.append(train_data_freq)
        eval_data_list.append(eval_data_freq)

    if DMTIME is True:
    
        model_2d_dm_time, score = frb_keras_convnet.construct_conv2d(
                        features_only=False, fit=True,
                        train_data=train_data_dm, eval_data=eval_data_dm, 
                        train_labels=train_labels, eval_labels=eval_labels,
                        epochs=5, nfilt1=32, nfilt2=64, 
                        nfreq=NDM, ntime=WIDTH)
    
        model_list.append(model_2d_dm_time)
        train_data_list.append(train_data_dm)
        eval_data_list.append(eval_data_dm)

    if TIME1D is True:

        model_1d_time, score = frb_keras_convnet.construct_conv1d(
                        features_only=False, fit=True,
                        train_data=train_data_1d, eval_data=eval_data_1d, 
                        train_labels=train_labels, eval_labels=eval_labels,
                        NTIME=64, nfilt1=64, nfilt2=128) 

        model_list.append(model_1d_time)
        train_data_list.append(train_data_1d)
        eval_data_list.append(eval_data_1d)

    if len(model_list)==1:
        score = model_list[0].evaluate(eval_data_list[0], eval_labels, batch_size=32)
        prob, predictions, mistakes = frb_keras_convnet.get_predictions(
                                model_list[0], eval_data_list[0], 
                                true_labels=eval_labels)
        print(mistakes)
        print(score)

    elif len(model_list)>1:
        print("Merging %d models" % len(model_list))

        model, score = frb_keras_convnet.merge_models(
                                         model_list, train_data_list, 
                                         train_labels, eval_data_list, eval_labels)

        prob, predictions, mistakes = frb_keras_convnet.get_predictions(
                                model, eval_data_list, 
                                true_labels=eval_labels)

        print(mistakes)
        print(score)

