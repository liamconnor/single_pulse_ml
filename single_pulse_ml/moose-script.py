import glob
import numpy as np

import matplotlib 
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import fit_model
import reader
import plotting-tools

# Data directory
#dir_name = '/home/connor/python_envs/2.7_L1mock/src/ch_L1mock/ch_L1mock/frb_incoherent_3b_triggers/200-525sim_ml/'
dir_name = '/home/connor/python_envs/2.7_L1mock/src/ch_L1mock/ch_L1mock/frb_incoherent_2b_triggers/200-525sim_ml_test/'
#dir_name = '/home/connor/python_envs/2.7_L1mock/src/ch_L1mock/ch_L1mock/frb_incoherent_2b_triggers/200-525sim_ml/'

array_name = 'DM' # Data array type. Either 'Freq' or 'DM' (freq/time vs. dm/time)
train_set = False # Creates training set if True, creates test set if False
run_predict = True # Applies saved fit, makes predictions on test data if True
DMsim = (376, 375) # Gives the DMs of simulated pulses
DMsim = (287,)

# Grab all the files in directory 
# with array type "array_name"
file_list = glob.glob('%sDM*%s*.npy' % (dir_name, array_name))

data_full, y = [], []

for fn in file_list:
    data = np.load(fn)
    data = reader.normalize_data(data)[200:400, :]
    data = reader.rebin_arr(data, 32, 250) 
    data_full.append(data)
    DM = fn.split('/')[-1].split('_')[0][2:]

    if int(DM) in DMsim:
        y.append(1)
    else:
        y.append(0)

data_full = np.concatenate(data_full, axis=0)
ndm, ntimes = data.shape
data_full.shape = (len(y), -1)
data_full[data_full!=data_full] = 0.0
y = np.array(y)

print data_full.shape

h, w = data.shape

plotting-tools.plot_gallery(data_arr, y, h, w, n_row=3, n_col=4)

print "\nData set has %d pulses %d nonpulses\n" \
        % (len(np.where(y==1)[0]), len(np.where(y==0)[0]))


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
    y_pred, class_report, conf_matrix = fit_model.predict_test(
                data_test, model, y_test=y_test, pca=pca)  










