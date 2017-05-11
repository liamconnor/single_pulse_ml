# To do: - include crab and b0329 pulses, but not to training set
#        - include other ML algorithms than just SVM
#

import sys

import glob
import numpy as np
import matplotlib # Allows for plotting with no $DISPLAY environment variable 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import fit_model
import reader
import plot_tools

assert len(sys.argv)==5, "Need four arguments: data_dir train_set run_predict array_name"

algorithm = 'SVM'

# Data directory

#dir_name = '/home/connor/python_envs/2.7_L1mock/src/ch_L1mock/ch_L1mock/frb_incoherent_1c_triggers/20-100sim_ml/' # B0329 directory
#dir_name = '/home/connor/python_envs/2.7_L1mock/src/ch_L1mock/ch_L1mock/frb_incoherent_3b_triggers/200-525sim_ml/'
#dir_name = '/home/connor/python_envs/2.7_L1mock/src/ch_L1mock/ch_L1mock/frb_incoherent_2b_triggers/200-525sim_ml_test/' # good test DM=287
#dir_name = '/home/connor/python_envs/2.7_L1mock/src/ch_L1mock/ch_L1mock/frb_incoherent_2b_triggers/200-525sim_ml/' # good train set

dir_name = sys.argv[1]
train_set = bool(int(sys.argv[2])) # Creates training set if True, creates test set if False
run_predict = bool(int(sys.argv[3])) # Applies saved fit, makes predictions on test data if True
array_name = sys.argv[4] # Data array type. Either 'Freq' or 'DM' (freq/time vs. dm/time)

assert array_name in ['DM', 'Freq'], "array_name should be in ['DM', 'Freq']"

DMsim = (376, 375) # Gives the DMs of simulated pulses
DMsim = (287,)#, 26, 376, 375)
#DMsim = (287,)
#DMsim = (26,)
plot = True
cmap_dict = {'DM': 'Greys', 'Freq': 'RdBu'}
nfreq, ntimes = 32, 250

def read_full_dir(dir_name, array_name='Freq', DMsim=(287, 26, 376, 375)):
    # Grab all the files in directory 
    # with array type "array_name

    file_list = glob.glob(dir_name)
    np.random.shuffle(file_list)
    data_full, y = [], []

    astart, aend = 0, -1

    if array_name is 'DM':
        astart, aend = 200, 400

    for fn in file_list:
        data = np.load(fn)
        data = reader.normalize_data(data)[astart:aend, :]
        data = reader.rebin_arr(data, nfreq, ntimes) 
        data_full.append(data)
        DM = fn.split('/')[-1].split('_')[0][2:]

        if int(DM) in DMsim:
            y.append(1)
        else:
            y.append(0)

    data_full = np.concatenate(data_full, axis=0)
    data_full.shape = (len(y), -1)
    data_full[data_full!=data_full] = 0.0
    y = np.array(y)

    print "\nTotal data set has %d pulses %d nonpulses\n" \
            % (len(np.where(y==1)[0]), len(np.where(y==0)[0]))

    return data_full, y

# Read in all files, combine into data array and label array
dir_full = '%sDM*%s*.npy' % (dir_name, array_name)
data_full, y = read_full_dir(dir_full, array_name=array_name, DMsim=DMsim)

# Split up into train and test data
X_train, X_test, y_train, y_test = train_test_split(data_full, y)

print "\nTraining set has %d pulses %d nonpulses\n" \
            % (len(np.where(y_train==1)[0]), len(np.where(y_train==0)[0]))
print "\nTest set has %d pulses %d nonpulses\n" \
            % (len(np.where(y_test==1)[0]), len(np.where(y_test==0)[0]))

if train_set is True:
    reader.write_data(X_train, y_train, './single_pulse_ml/data/training_data_pf%s.npy' % array_name)

    if algorithm is 'SVM':
        model, pca = fit_model.fit_svm('./single_pulse_ml/data/training_data_pf%s.npy' % array_name)
        reader.write_pkl(pca, './single_pulse_ml/model/training_data_pf_pca%s' % array_name)

    elif algorithm is 'kneighbors':
        model = fit_model.fit_kneighbors('./single_pulse_ml/data/training_data_pf%s.npy' % array_name)    

    reader.write_pkl(model, './single_pulse_ml/model/training_data_pf_%s%s' % (algorithm, array_name))

if train_set is False:
    reader.write_data(X_test, y_test, 'test_data_pf%s.npy' % array_name)

if run_predict is True:
    model = reader.read_pkl('./single_pulse_ml/model/training_data_pf_%s%s.pkl' % (algorithm, array_name))

    if algorithm is 'SVM':
        pca = reader.read_pkl('./single_pulse_ml/model/training_data_pf_pca%s.pkl' % array_name)
    else:
        pca = None 

    data_test, y_test = reader.read_data('test_data_pf%s.npy' % array_name)
    y_pred, class_report, conf_matrix = fit_model.predict_test(
                data_test, model, y_test=y_test, pca=pca)  

if plot:
    target_names = np.array(['RFI', 'Pulse'], dtype='|S17')

    if train_set:
        figname = './single_pulse_ml/plots/%s_train.png' % array_name
        prediction_titles = plot_tools.get_title(y, target_names)
        print "Plotting training set to file: %s" % figname

    elif train_set is False and run_predict is True:

        figname = './single_pulse_ml/plots/%s_test.png' % array_name 
        pred_name = plot_tools.get_title(y_pred.astype(int), target_names)
        true_name = plot_tools.get_title(y_pred.astype(int), target_names)
        prediction_titles = ['predicted: %s\ntrue:      %s' % (pred_name[ii], true_name[ii])     
                        for ii in range(len(pred_name))]
        print "Plotting test set to file: %s\n" % figname

    if train_set is True:
        suptitle = 'Training Data'
    else:
        suptitle = 'Test Data'

    plot_tools.plot_gallery(data_full, prediction_titles, 
                      nfreq, ntimes, n_row=8, n_col=8, figname=figname, 
                      cmap=cmap_dict[array_name], suptitle=suptitle)








