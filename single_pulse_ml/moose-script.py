import glob
import numpy as np

import matplotlib 
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import fit_model
import reader

# Data directory
dir_name = '/home/connor/python_envs/2.7_L1mock/src/ch_L1mock/ch_L1mock/frb_incoherent_3b_triggers/200-525sim_ml/'
#dir_name = '/home/connor/python_envs/2.7_L1mock/src/ch_L1mock/ch_L1mock/frb_incoherent_2b_triggers/200-525sim_ml_test/'
#dir_name = '/home/connor/python_envs/2.7_L1mock/src/ch_L1mock/ch_L1mock/frb_incoherent_2b_triggers/200-525sim_ml/'


array_name = 'Freq' # Data array type. Either 'Freq' or 'DM' (freq/time vs. dm/time)
train_set = False # Creates training set if True, creates test set if False
run_predict = True # Applies saved fit, makes predictions on test data if True
DMsim = (376, 375)
#DMsim = (287,)

fl = glob.glob('%sDM*%s*.npy' % (dir_name, array_name))

data_full, y = [], []

for ff in fl:
    data = np.load(ff)
    data = reader.normalize_data(data)
    data = reader.rebin_arr(data, 32, 250) 
    data_full.append(data)
    DM = ff.split('/')[-1].split('_')[0][2:]
    print ff
    if int(DM) in DMsim:
        y.append(1)
    else:
        y.append(0)

data_full = np.concatenate(data_full, axis=0)
ndm, ntimes = data.shape
data_full.shape = (len(y), -1)
data_full[data_full!=data_full] = 0.0
y = np.array(y)

if train_set is True:
    reader.write_training_data(data_full, y, 'training_data_pf%s.npy' % array_name)
    model, pca = fit_model.fit_svm('training_data_pf%s.npy' % array_name)
    reader.write_pkl(pca, 'training_data_pf_pca%s' % array_name)
    reader.write_pkl(model, 'training_data_pf_model%s' % array_name)

if train_set is False:
    reader.write_training_data(data_full, y, 'test_data_pf%s.npy' % array_name)

if run_predict is True:
    model = reader.read_pkl('training_data_pf_model%s.pkl' % array_name)
    pca = reader.read_pkl('training_data_pf_pca%s.pkl' % array_name)
    data_test, y_test = reader.read_training_data('test_data_pf%s.npy' % array_name)
    fit_model.predict_test(data_test, model, y_test=y_test, pca=pca)  




"""
data_test, y_test = [],[]

counter = 0

for ii, ff in enumerate(fl_test):
    data_t = np.load(ff)
    data_t = reader.normalize_data(data_t)
    data_t = reader.rebin_arr(data_t, 64, 250)
    data_t = pca.transform(data_t.reshape(-1))
    data_test.append(data_t)

    yp, DM = model.predict(data_t)[0], int(ff.split('/')[-1].split('_')[0][2:])

    if DM==287:
        yt=1
        y_test.append(1)
    else:
        yt=0
        y_test.append(0)

    print "pred test correct? dm"
    print yp, yt, abs(yp-yt), DM

    counter += abs(yp-yt)
    print ii, counter

print counter / float(ii)

data_test = np.concatenate(data_test, axis=0)
ndm, ntimes = data_t.shape
data_test.shape = (-1, ndm*ntimes)
print y_test
print data_test.shape

fit_model.predict_test(data_test, model, y_test=y_test, pca=None)
"""
