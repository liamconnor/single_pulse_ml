""" Tools for building and training deep neural
    networks in keras using the tensorflow backend.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
from numpy.random import seed
import h5py

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Merge
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPooling2D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.models import load_model


def get_predictions(model, data, true_labels=None):
    """ Take a keras.model object, a data array, 
    and true_labels, and return the probability of 
    each feature being a TP, the prediction itself, 
    and the mistakes.
    """
    if len(true_labels.shape)==2:
        true_labels = true_labels[:,1]

    prob = model.predict(data)
    predictions = np.round(prob[:, 1])

    if true_labels is not None:
        mistakes = np.where(predictions!=true_labels)[0]
    else:
        mistakes = []

    return prob, predictions, mistakes

def get_classification_results(y_true, y_pred):
    """ Take true labels (y_true) and model-predicted 
    label (y_pred) for a binary classifier, and return 
    true_positives, false_positives, true_negatives, false_negatives
    """

    true_positives = np.where((y_true==1) & (y_pred==1))[0]
    false_positives = np.where((y_true==0) & (y_pred==1))[0]
    true_negatives = np.where((y_true==0) & (y_pred==0))[0]
    false_negatives = np.where((y_true==1) & (y_pred==0))[0]

    return true_positives, false_positives, true_negatives, false_negatives

def confusion_mat(y_true, y_pred):
    """ Generate a confusion matrix for a 
    binary classifier based on true labels (
    y_true) and model-predicted label (y_pred)
    
    returns np.array([[TP, FP],[FN, TN]])
    """
    TP, FP, TN, FN = get_classification_results(y_true, y_pred)

    NTP = len(TP)
    NFP = len(FP)
    NTN = len(TN)
    NFN = len(FN)

    conf_mat = np.array([[NTP, NFP],[NFN, NTN]])

    return conf_mat

def print_metric(y_true, y_pred):
    """ Take true labels (y_true) and model-predicted 
    label (y_pred) for a binary classifier
    and print a confusion matrix, metrics, 
    return accuracy, precision, recall, fscore
    """
    conf_mat = confusion_mat(y_true, y_pred)

    NTP, NFP, NTN, NFN = conf_mat[0,0], conf_mat[0,1], conf_mat[1,1], conf_mat[1,0]

    print("Confusion matrix: %s" % conf_mat)
    accuracy = float(NTP + NTN)/conf_mat.sum()
    precision = float(NTP) / (NTP + NFP)
    recall = float(NTP) / (NTP + NFN)
    fscore = 2*precision*recall/(precision+recall)

    print("accuracy: %f" % accuracy)
    print("precision: %f" % precision)
    print("recall: %f" % recall)
    print("fscore: %f" % fscore)

    return accuracy, precision, recall, fscore

def split_data(fn, NFREQ=16, NTIME=250, train_size=0.75):
    """ Read in numpy file and split randomly into 
    train and test data 

    Parameters:
    ----------
    fn : str 
        file name
    train_size : np.float 
        fraction of data to train on 

    Return: 
    ------
    train_data: np.array

    eval_data: np.array

    train_labels: np.array

    eval_labels: np.array
    """
    f = np.load(fn)

    train_data, eval_data, train_labels, eval_labels = \
              train_test_split(f[:, :-1], f[:, -1], train_size=train_size)

    train_data = train_data[..., None].reshape(-1, NFREQ, NTIME, 1)
    eval_data = eval_data[..., None].reshape(-1, NFREQ, NTIME, 1)

    return train_data, eval_data, train_labels, eval_labels


def construct_ff1d(features_only=False, fit=False, 
                     train_data=None, train_labels=None,
                     eval_data=None, eval_labels=None, 
                     nbeam=32, epochs=5,
                     nlayer1=32, nlayer2=64, batch_size=32):
    """ Build a one-dimensional feed forward neural network 
    with a binary classifier. Can be used for, e.g.,
    multi-beam detections. 

    Parameters:
    ----------
    features_only : bool 
        Don't construct full model, only features layers 
    fit : bool 
        Fit model 
    train_data : ndarray
        (ntrain, ntime, 1) float64 array with training data
    train_labels :  ndarray
        (ntrigger, 2) binary labels of training data [0, 1] = FRB, [1, 0]=RFI 
    eval_data : ndarray
        (neval, ntime, 1) float64 array with evaluation data
    eval_labels : 
        (neval, 2) binary labels of eval data 
    nbeam : int 
        Number of input beams (more generally, number of data inputs) 
    epochs : int 
        Number of training epochs 
    nlayer1 : int
        Number of neurons in first hidden layer 
    nlayer2 : int 
        Number of neurons in second hidden layer 
    batch_size : int 
        Number of batches for training   

    Returns
    -------
    model : XX

    score : XX

    """
    model = Sequential()
    model.add(Dense(nlayer1, input_dim=nbeam, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(nlayer2, init='normal', activation='relu'))

    if features_only is True:
        model.add(BatchNormalization()) # hack
        return model, []

    model.add(Dropout(0.4))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                       optimizer='rmsprop',
                       metrics=['accuracy'])

    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)
    score = model.evaluate(eval_data, eval_labels, batch_size=batch_size)

    return model, score

def construct_conv2d(features_only=False, fit=False, 
                     train_data=None, train_labels=None,
                     eval_data=None, eval_labels=None, 
                     nfreq=16, ntime=250, epochs=5,
                     nfilt1=32, nfilt2=64, batch_size=32):
    """ Build a two-dimensional convolutional neural network
    with a binary classifier. Can be used for, e.g.,
    freq-time dynamic spectra of pulsars, dm-time intensity array.

    Parameters:
    ----------
    features_only : bool 
        Don't construct full model, only features layers 
    fit : bool 
        Fit model 
    train_data : ndarray
        (ntrain, ntime, 1) float64 array with training data
    train_labels :  ndarray
        (ntrigger, 2) binary labels of training data [0, 1] = FRB, [1, 0]=RFI 
    eval_data : ndarray
        (neval, ntime, 1) float64 array with evaluation data
    eval_labels : 
        (neval, 2) binary labels of eval data 
    epochs : int 
        Number of training epochs 
    nfilt1 : int
        Number of neurons in first hidden layer 
    nfilt2 : int 
        Number of neurons in second hidden layer 
    batch_size : int 
        Number of batches for training   

    Returns
    -------
    model : XX

    score : np.float 
        accuracy, i.e. fraction of predictions that are correct 

    """

    if train_data is not None:
        nfreq=train_data.shape[1]
        ntime=train_data.shape[2]

    model = Sequential()
    # this applies 32 convolution filters of size 5x5 each.
    model.add(Conv2D(nfilt1, (5, 5), activation='relu', input_shape=(nfreq, ntime, 1)))

    #model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Randomly drop some fraction of nodes (set weights to 0)
    model.add(Dropout(0.4)) 
    model.add(Conv2D(nfilt2, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4)) 
    model.add(Flatten())

    if features_only is True:
        model.add(BatchNormalization()) # hack
        return model, [] #hack

    model.add(Dense(256, activation='relu')) # should be 1024 hack

#    model.add(Dense(1024, activation='relu')) # remove for now hack
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # train_labels = keras.utils.to_categorical(train_labels)
    # eval_labels = keras.utils.to_categorical(eval_labels)

    if fit is True:
        print("Using batch_size: %d" % batch_size)
        print("Using %d epochs" % epochs)
        cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, 
                                         batch_size=32, write_graph=True, write_grads=False, 
                                         write_images=True, embeddings_freq=0, embeddings_layer_names=None, 
                                         embeddings_metadata=None)

        model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, callbacks=[cb])
        score = model.evaluate(eval_data, eval_labels, batch_size=batch_size)
        print("Conv2d only")
        print(score)

    return model, score

def construct_conv1d(features_only=False, fit=False, 
                     train_data=None, train_labels=None,
                     eval_data=None, eval_labels=None,
                     nfilt1=64, nfilt2=128,
                     batch_size=16, epochs=5):
    """ Build a one-dimensional convolutional neural network
    with a binary classifier. Can be used for, e.g.,
    pulse profiles. 

    Parameters:
    ----------
    features_only : bool 
        Don't construct full model, only features layers 
    fit : bool 
        Fit model 
    train_data : ndarray
        (ntrain, ntime, 1) float64 array with training data
    train_labels :  ndarray
        (ntrigger, 2) binary labels of training data [0, 1] = FRB, [1, 0]=RFI 
    eval_data : ndarray
        (neval, ntime, 1) float64 array with evaluation data
    eval_labels : 
        (neval, 2) binary labels of eval data 
    epochs : int 
        Number of training epochs 
    nfilt1 : int
        Number of neurons in first hidden layer 
    nfilt2 : int 
        Number of neurons in second hidden layer 
    batch_size : int 
        Number of batches for training   

    Returns
    -------
    model : XX

    score : XX

    """

    if train_data is not None:
        NTIME=train_data.shape[1]

    model = Sequential()
    model.add(Conv1D(nfilt1, 3, activation='relu', input_shape=(NTIME, 1)))
    model.add(Conv1D(nfilt1, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(nfilt2, 3, activation='relu'))
    model.add(Conv1D(nfilt2, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())

    if features_only is True:
        return model

    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])

    if fit is True:
        model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)
        score = model.evaluate(eval_data, eval_labels, batch_size=16)
        print("Conv1d only")
        print(score)

    return model, score


def merge_models(model_list, train_data_list, 
                 train_labels, eval_data_list, eval_labels,
                 batch_size=32, epochs=5):
    """ Take list of models, list of training data, 
    merge models and train as a single network. 
    """
    

    model = Sequential()
    model.add(Merge(model_list, mode = 'concat'))
    #model.add(Dense(256, activation='relu'))
    model.add(Dense(2, init = 'normal', activation = 'sigmoid'))
    sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
    model.compile(loss = 'binary_crossentropy', 
          optimizer=sgd, 
          metrics=['accuracy'])
    seed(2017)
    model.fit(train_data_list, train_labels, 
                    batch_size=batch_size, nb_epoch=epochs, verbose=1)
    score = model.evaluate(eval_data_list, eval_labels, batch_size=batch_size)

    return model, score


def read_hdf5(fn):
    f = h5py.File(fn, 'r')
    data_freq = f['data_freq_time'][:]
    y = f['labels'][:]

    try:
        data_dm = f['data_dm_time'][:]
    except:
        print("dm-time dataset not there")
        data_dm = None

    return data_freq, y, data_dm

