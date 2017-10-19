from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
from numpy.random import seed

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Merge
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPooling2D, MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split


def split_data(fn, train_size=0.75):
    fn = './data/_data_nt250_nf16_dm0_snrmax200.npy'
    f = np.load(fn)

    train_data, eval_data, train_labels, eval_labels = \
              train_test_split(f[:, :-1], f[:, -1], train_size=train_size)

    train_data = train_data[..., None].reshape(-1, 16, 250, 1)
    eval_data = eval_data[..., None].reshape(-1, 16, 250, 1)

    return train_data, eval_data, train_labels, eval_labels


def construct_conv2d(features_only=False, fit=False, 
                     train_data=None, train_labels=None,
                     eval_data=None, eval_labels=None, 
                     nfreq=16, ntime=250):

    if train_data is not None:
        nfreq=train_data.shape[1]
        ntime=train_data.shape[2]

    model = Sequential()
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(nfreq, ntime, 1)))
    #model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))

    if features_only is True:
        return model

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    if fit is True:
        model.fit(train_data, train_labels, batch_size=32, epochs=10)
        score = model.evaluate(eval_data, eval_labels, batch_size=32)
        print("Conv2d only")
        print(score)

    return model 

def construct_conv1d(features_only=False, fit=False, 
                     train_data=None, train_labels=None,
                     eval_data=None, eval_labels=None):

    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(250, 1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())

    if features_only is True:
        return model

    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])

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

if __name__=='__main__':
    width=32
    nfreq=16
    ntime=250
    tl, th = ntime//2-width, ntime//2+width

    fn = './data/_data_nt250_nf16_dm0_snrmax100.npy'

    if len(sys.argv) > 1:
        fn = sys.argv[1]

    train_data, eval_data, train_labels, eval_labels = split_data(fn, train_size=0.75)

    train_data_1d = train_data.mean(1)
    eval_data_1d = eval_data.mean(1)

    right_branch_2d = construct_conv2d(features_only=False, fit=True,
                            train_data=train_data[:,:,tl:th], eval_data=eval_data[:,:,tl:th], 
                            train_labels=train_labels, eval_labels=eval_labels)

    left_branch_1d = construct_conv1d(features_only=False, fit=True,
                            train_data=train_data_1d, eval_data=eval_data_1d, 
                            train_labels=train_labels, eval_labels=eval_labels)


    model = merge_models(left_branch_1d, right_branch_2d)

    seed(2017)
    model.fit([train_data_1d, train_data], train_labels, 
        batch_size = 2000, nb_epoch = 10, verbose = 1)
    score = model.evaluate([eval_data_1d, eval_data], eval_labels, batch_size=32)
    print(score)



#   seed(2017)
#   model.fit([X1, X2], Y.values, batch_size = 2000, nb_epoch = 100, verbose = 1)

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


