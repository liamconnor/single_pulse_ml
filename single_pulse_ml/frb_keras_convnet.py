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


def construct_conv2d(features_only=False, fit=False, 
                     train_data=None, train_labels=None,
                     eval_data=None, eval_labels=None, 
                     nfreq=16, ntime=250, epochs=5,
                     nfilt1=32, nfilt2=64):

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

    train_labels = keras.utils.to_categorical(train_labels)
    eval_labels = keras.utils.to_categorical(eval_labels)

    if fit is True:
        model.fit(train_data, train_labels, batch_size=32, epochs=epochs)
        score = model.evaluate(eval_data, eval_labels, batch_size=32)
        print("Conv2d only")
        print(score)

    return model 

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

    train_labels = keras.utils.to_categorical(train_labels)
    eval_labels = keras.utils.to_categorical(eval_labels)

    if fit is True:
        model.fit(train_data, train_labels, batch_size=16, epochs=5)
        score = model.evaluate(eval_data, eval_labels, batch_size=16)
        print("Conv1d only")
        print(score)

    return model

def merge_models(left_branch, right_branch):
    # Configure the accuracy metric for evaluation
    metrics = ["accuracy", "precision", "false_negatives", "recall"] 

    model = Sequential()
    model.add(Merge([left_branch, right_branch], mode = 'concat'))
    #model.add(Dense(256, activation='relu'))
    model.add(Dense(1, init = 'normal', activation = 'sigmoid'))
    sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
    model.compile(loss = 'binary_crossentropy', 
                  optimizer=sgd, 
                  metrics=['accuracy'])

    return model

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

if __name__=='__main__':

    fn = './data/_data_nt250_nf16_dm0_snrmax100.npy'

    if len(sys.argv) > 1:
        fn = sys.argv[1]
    
    print("Using %s" % fn)

    NDM=150
    NFREQ=32
    NTIME=250
    WIDTH=64
    tl, th = NTIME//2-WIDTH//2, NTIME//2+WIDTH//2
    train_size=0.75

    ftype = fn.split('.')[-1]
    print(ftype)
    if ftype=='hdf5':

        data_freq, y, data_dm = read_hdf5(fn)
        data_freq = data_freq[..., tl:th]
        data_dm = data_dm[..., tl:th]
        
        # tf expects 4D tensors
        data_dm = data_dm[..., None]
        data_freq = data_freq[..., None]
        data_1d = data_freq.mean(1)

        NTRIGGER = len(y)
        NTRAIN = int(train_size * NTRIGGER)
        train_size = 0.25

        ind = np.arange(NTRIGGER)
        np.random.shuffle(ind)

        ind_train = ind[:NTRAIN]
        ind_eval = ind[NTRAIN:]

        train_data_dm, eval_data_dm = data_dm[ind_train], data_dm[ind_eval]
        train_data_freq, eval_data_freq = data_freq[ind_train], data_freq[ind_eval]
        train_data_1d, eval_data_1d = data_1d[ind_train], data_1d[ind_eval]

        train_labels, eval_labels = y[ind_train], y[ind_eval]
        
        model_2d_freq_time = construct_conv2d(features_only=False, fit=True,
                        train_data=train_data_freq, eval_data=eval_data_freq, 
                        train_labels=train_labels, eval_labels=eval_labels,
                        epochs=5, nfilt1=32, nfilt2=64, 
                        nfreq=NFREQ, ntime=WIDTH)

        model_2d_dm_time = construct_conv2d(features_only=False, fit=True,
                        train_data=train_data_dm, eval_data=eval_data_dm, 
                        train_labels=train_labels, eval_labels=eval_labels,
                        epochs=5, nfilt1=32, nfilt2=64, 
                        nfreq=NDM, ntime=WIDTH)
        
        model_1d_time = construct_conv1d(features_only=False, fit=True,
                            train_data=train_data_1d, eval_data=eval_data_1d, 
                            train_labels=train_labels, eval_labels=eval_labels,
                            NTIME=64, nfilt1=64, nfilt2=128) 

        # Configure the accuracy metric for evaluation
        metrics = ["accuracy", "precision", "false_negatives", "recall"] 

        model = Sequential()
        model.add(Merge([model_2d_freq_time, model_2d_dm_time, model_1d_time], mode = 'concat'))
        #model.add(Dense(256, activation='relu'))
        model.add(Dense(1, init = 'normal', activation = 'sigmoid'))
        sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
        model.compile(loss = 'binary_crossentropy', 
                  optimizer=sgd, 
                  metrics=['accuracy'])

        seed(2017)

        print("Now merging models")
        model.fit([train_data_freq, train_data_dm, train_data_1d], train_labels, 
                    batch_size = 2000, nb_epoch = 5, verbose = 1)

        score = model.evaluate([eval_data_freq, eval_data_dm, eval_data_1d], 
                                eval_labels, batch_size=32)

        prob, predictions, mistakes = get_predictions(
                                model, [eval_data_freq, eval_data_dm, eval_data_1d], 
                                true_labels=eval_labels)
        print(mistakes)


    elif ftype is 'npy':
        train_data, eval_data, train_labels, eval_labels = \
                    split_data(fn, train_size=0.25, NFREQ=NFREQ, NTIME=NTIME,)

    else:
        print("Input file type not recognized")
        raise 


    # train_data = train_data[:,:,tl:th]
    # eval_data = eval_data[:,:,tl:th]

    # train_data_1d = train_data.mean(1)
    # eval_data_1d = eval_data.mean(1)

    # right_branch_2d = construct_conv2d(features_only=False, fit=True,
    #                         train_data=train_data, eval_data=eval_data, 
    #                         train_labels=train_labels, eval_labels=eval_labels,
    #                         epochs=5, nfilt1=32, nfilt2=64, nfreq=NFREQ, ntime=WIDTH)

    # left_branch_1d = construct_conv1d(features_only=True, fit=True,
    #                         train_data=train_data_1d, eval_data=eval_data_1d, 
    #                         train_labels=train_labels, eval_labels=eval_labels,
    #                         NTIME=64, nfilt1=64, nfilt2=128)   

    # model = merge_models(left_branch_1d, right_branch_2d)

    # seed(2017)
    # model.fit([train_data_1d, train_data], train_labels, 
    #     batch_size = 2000, nb_epoch = 10, verbose = 1)
    # score = model.evaluate([eval_data_1d, eval_data], 
    #                         eval_labels, batch_size=32)

    # print(score)

    # prob, predictions, mistakes = get_predictions(
    #                             model, [eval_data_1d, eval_data], 
    #                             true_labels=eval_labels)


#   seed(2017)
#   model.fit([X1, X2], Y.values, batch_size = 2000, nb_epoch = 100, verbose = 1)

# for ii in range(len(mis)):
#     subplot(7,7,ii+1)
#     imshow(eval_data[mis[ii],:,:,0],aspect='auto',interpolation='nearest',cmap='Greys')
#     title(str(predm[mis[ii]]))
#     axis('off')

# Junk Code that might not be junk.
# Generate dummy data
# x_train = np.random.random((100, 100, 100, 3))
# y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
# x_test = np.random.random((20, 100, 100, 3))
# y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

# fn = './data/_data_nt250_nf16_dm0_snrmax200.npy'
# f = np.load(fn)
# d, y = f[:, :-1], f[:, -1]

# d_train, y_train = (f[::10, :-1]).astype(np.float32), (f[::10, -1]).astype(np.int32)
# d_test, y_test = (f[1::10, :-1]).astype(np.float32), (f[1::10, -1]).astype(np.int32)

# d_train = d_train[..., None].reshape(-1, 16, 250, 1)
# d_test = d_test[..., None].reshape(-1, 16, 250, 1)

# # Turn these into categorical vectors
# y_train_cat = keras.utils.to_categorical(y_train)
# y_test_cat = keras.utils.to_categorical(y_test)

# train_data, eval_data, train_labels, eval_labels = \
#               train_test_split(d, y, train_size=0.75)

# train_data = train_data.reshape(-1, nfreq, ntime)[..., None]
# eval_data = eval_data.reshape(-1, nfreq, ntime)[..., None]


