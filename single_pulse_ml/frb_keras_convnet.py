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
from keras.layers import MaxPooling2D, MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.models import load_model

def load_model(fn):
    model = load_model(fn)

    return model


def get_predictions(model, data, true_labels=None):
    prob = model.predict(data)
    predictions = np.round(prob[:, 0])

    if true_labels is not None:
        mistakes = np.where(predictions!=true_labels)[0]
    else:
        mistakes = []

    return prob, predictions, mistakes


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
        with a binary classifier. Can be used for 
        multi-beam detections. 
    """
    model = Sequential()
    model.add(Dense(nlayer1, input_dim=nbeam, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(nlayer2, init='normal', activation='relu'))
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

    if train_data is not None:
        nfreq=train_data.shape[1]
        ntime=train_data.shape[2]

    model = Sequential()
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(nfilt1, (5, 5), activation='relu', input_shape=(nfreq, ntime, 1)))

    #model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Randomly drop some fraction of nodes (set weights to 0)
    model.add(Dropout(0.4))

    model.add(Conv2D(nfilt2, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(256, activation='relu')) # should be 1024 hack

    if features_only is True:
        return model

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
        # cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, 
        #                                  batch_size=32, write_graph=True, write_grads=False, 
        #                                 write_images=False, embeddings_freq=0, embeddings_layer_names=None, 
        #                                 embeddings_metadata=None)

        model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)#, callbacks=[cb])
        score = model.evaluate(eval_data, eval_labels, batch_size=batch_size)
        print("Conv2d only")
        print(score)

    return model, score

def construct_conv1d(features_only=False, fit=False, 
                     train_data=None, train_labels=None,
                     eval_data=None, eval_labels=None,
                     NTIME=250, nfilt1=64, nfilt2=128):

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

    # train_labels = keras.utils.to_categorical(train_labels)
    # eval_labels = keras.utils.to_categorical(eval_labels)

    if fit is True:
        model.fit(train_data, train_labels, batch_size=16, epochs=5)
        score = model.evaluate(eval_data, eval_labels, batch_size=16)
        print("Conv1d only")
        print(score)

    return model, score

# def merge_models(left_branch, right_branch):
#     # Configure the accuracy metric for evaluation
#     metrics = ["accuracy", "precision", "false_negatives", "recall"] 

#     model = Sequential()
#     model.add(Merge([left_branch, right_branch], mode = 'concat'))
#     #model.add(Dense(256, activation='relu'))
#     model.add(Dense(1, init = 'normal', activation = 'sigmoid'))
#     sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
#     model.compile(loss = 'binary_crossentropy', 
#                   optimizer=sgd, 
#                   metrics=['accuracy'])

#     return model

def merge_models(model_list, train_data_list, 
                 train_labels, eval_data_list, eval_labels):

    model = Sequential()
    model.add(Merge(model_list, mode = 'concat'))
    #model.add(Dense(256, activation='relu'))
    model.add(Dense(2, init = 'normal', activation = 'sigmoid'))
    sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
    model.compile(loss = 'binary_crossentropy', 
          optimizer=sgd, 
          metrics=['accuracy'])
    model.fit(train_data_list, train_labels, 
                    batch_size = 2000, nb_epoch = 5, verbose = 1)
    score = model.evaluate(eval_data_list, eval_labels, batch_size=32)

    return model, score

def merge_models_three(left_branch, right_branch):
    # Configure the accuracy metric for evaluation
    metrics = ["accuracy", "precision", "false_negatives", "recall"] 

    model = Sequential()
    model.add(Merge([left_branch, right_branch, left_branch], mode = 'concat'))
    #model.add(Dense(256, activation='relu'))
    model.add(Dense(1, init = 'normal', activation = 'sigmoid'))
    sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
    model.compile(loss = 'binary_crossentropy', 
                  optimizer=sgd, 
                  metrics=['accuracy'])

    return model

def run_2dconv_freq_time(train_data, train_labels, eval_data=None, 
                         eval_labels=None, 
                         NFREQ=16, NTIME=250, WIDTH=64,
                         epochs=5, nfilt1=32, nfilt2=64, 
                         features_only=False):
    """ Data array should be (NTRIGGER, NFREQ, NTIME)
    """
    assert(len(train_data.shape)==3)

    train_data = train_data.reshape(-1, NFREQ, NTIME, 1)

    if eval_data is not None:
        assert(len(eval_data.shape)==3)
        eval_data = eval_data.reshape(-1, NFREQ, NTIME, 1)

    model_2d_freq_time = construct_conv2d(features_only=features_only, fit=True,
                        train_data=train_data, eval_data=eval_data, 
                        train_labels=train_labels, eval_labels=eval_labels,
                        epochs=epochs, nfilt1=nfilt1, nfilt2=nfilt2, 
                        nfreq=NFREQ, ntime=WIDTH)

    return model_2d_freq_time

def run_2dconv_dm_time():
    pass

def run_1dconv_time():
    pass

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

