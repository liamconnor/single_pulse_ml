# Liam Connor 25 July 2018
# Script to classify single-pulses 
# using tensorflow/keras model. Output probabilities 
# can be saved and plotted

import optparse
import numpy as np
import h5py

import frbkeras
import reader
import plot_tools

def classify(data, model, save_ranked=False, 
             plot_ranked=False, prob_threshold=0.5,
             fnout='ranked'):

    model = frbkeras.load_model(fn_model_freq)

    dshape = data.shape

    # normalize data
    data = data.reshape(len(data), -1)
    data -= np.median(data, axis=-1)[:, None]
    data /= np.std(data, axis=-1)[:, None]

    # zero out nans
    data[data!=data] = 0.0
    data = data.reshape(dshape)

    if data_freq.shape[-1]!=1:
        data = data[..., None]

    if len(model.input.shape)==3:
        data = data.mean(1)
        
    y_pred_prob = model.predict(data)
    y_pred_prob = y_pred_prob[:,1]

    ind_frb = np.where(y_pred_prob>prob_threshold)[0]

    print("\n%d out of %d events with probability > %.2f:\n %s" % 
            (len(ind_frb), len(y_pred_prob), 
                prob_threshold, ind_frb))

    low_to_high_ind = np.argsort(y_pred_prob)

    if save_ranked is True:
        print("Need to fix the file naming")
        fnout_ranked = fn_data.rstrip('.hdf5') + \
                       'freq_time_candidates.hdf5'

        g = h5py.File(fnout_ranked, 'w')
        g.create_dataset('data_frb_candidate', data=data[ind_frb])
        g.create_dataset('frb_index', data=ind_frb)
        g.create_dataset('probability', data=y_pred_prob)
        g.close()
        print("\nSaved them and all probabilities to: \n%s" % fnout_ranked)

    if plot_ranked is True:
        if save_ranked is False:
            argtup = (data[ind_frb], ind_frb, y_pred_prob)

            plot_tools.plot_multiple_ranked(argtup, nside=10, \
                                            fnfigout=fnout, ascending=False)
        else:
            plot_tools.plot_multiple_ranked(fnout_ranked, nside=10, \
                                            fnfigout=fnout, ascending=False)


if __name__=="__main__":
    parser = optparse.OptionParser(prog="classify.py", \
                        version="", \
                        usage="%prog FN_DATA FN_MODEL [OPTIONS]", \
                        description="Apply DNN model to FRB candidates")

    parser.add_option('--fn_model_dm', dest='fn_model_dm', type='str', \
                        help="Filename of dm_time model. Default None", \
                        default=None)

    parser.add_option('--fn_model_time', dest='fn_model_time', type='str', \
                        help="Filename of 1d time model. Default None", \
                        default=None)

    parser.add_option('--fn_model_mb', dest='fn_model_mb', type='str', \
                        help="Filename of multibeam model. Default None", \
                        default=None)

    parser.add_option('--pthresh', dest='prob_threshold', type='float', \
                        help="probability treshold", default=0.5)

    parser.add_option('--save_ranked', dest='save_ranked', 
                        action='store_true', \
                        help="save FRB events + probabilities", \
                        default=False)

    parser.add_option('--plot_ranked', dest='plot_ranked', \
                        action='store_true',\
                        help="plot triggers", default=False)

    parser.add_option('--twindow', dest='twindow', type='int', \
                        help="time width, default 64", default=64)

    parser.add_option('--fnout', dest='fnout', type='str', \
                       help="beginning of figure names", \
                       default='ranked_trig')

    options, args = parser.parse_args()

    assert len(args)==2, "Arguments are FN_DATA FN_MODEL [OPTIONS]"

    fn_data = args[0]
    fn_model_freq = args[1]

    print("Using datafile %s" % fn_data)
    print("Using keras model in %s" % fn_model_freq)

    data_freq, y, data_dm, data_mb = reader.read_hdf5(fn_data)

    NFREQ = data_freq.shape[1]
    NTIME = data_freq.shape[2]
    WIDTH = options.twindow

    # low time index, high time index
    tl, th = NTIME//2-WIDTH//2, NTIME//2+WIDTH//2

    if data_freq.shape[-1] > (th-tl):
        data_freq = data_freq[..., tl:th]

    classify(data_freq, fn_model_freq, 
             save_ranked=options.save_ranked, 
             plot_ranked=options.plot_ranked, 
             prob_threshold=options.prob_threshold,
             fnout=options.fnout)

    if options.fn_model_dm is not None:
        if len(data_dm)>0:
            classify(data_dm, options.fn_model_dm, 
             save_ranked=options.save_ranked, 
             plot_ranked=options.plot_ranked, 
             prob_threshold=options.prob_threshold,
             fnout=options.fnout)

    if options.fn_model_time is not None:
        classify(data_freq, options.fn_model_time, 
             save_ranked=options.save_ranked, 
             plot_ranked=options.plot_ranked, 
             prob_threshold=options.prob_threshold,
             fnout=options.fnout)

    if options.fn_model_mb is not None:
        classify(data_mb, options.fn_model_mb, 
             save_ranked=options.save_ranked, 
             plot_ranked=options.plot_ranked, 
             prob_threshold=options.prob_threshold,
             fnout=options.fnout)

    exit()

    dshape = data_freq.shape

    # normalize data
    data_freq = data_freq.reshape(len(data_freq), -1)
    data_freq -= np.median(data_freq, axis=-1)[:, None]
    data_freq /= np.std(data_freq, axis=-1)[:, None]

    # zero out nans
    data_freq[data_freq!=data_freq] = 0.0
    data_freq = data_freq.reshape(dshape)

    if data_freq.shape[-1]!=1:
        data_freq = data_freq[..., None]

    model = frbkeras.load_model(fn_model_freq)

    if len(model.input.shape)==3:
        data_freq = data_freq.mean(1)
        
    y_pred_prob = model.predict(data_freq)
    y_pred_prob = y_pred_prob[:,1]

    ind_frb = np.where(y_pred_prob>options.prob_threshold)[0]

    print("\n%d out of %d events with probability > %.2f:\n %s" % 
            (len(ind_frb), len(y_pred_prob), 
                options.prob_threshold, ind_frb))

    low_to_high_ind = np.argsort(y_pred_prob)

    if options.save_ranked is True:
        fnout_ranked = fn_data.rstrip('.hdf5') + 'freq_time_candidates.hdf5'

        g = h5py.File(fnout_ranked, 'w')
        g.create_dataset('data_frb_candidate', data=data_freq[ind_frb])
        g.create_dataset('frb_index', data=ind_frb)
        g.create_dataset('probability', data=y_pred_prob)
        g.close()
        print("\nSaved them and all probabilities to: \n%s" % fnout_ranked)

    if options.plot_ranked is True:
        if options.save_ranked is False:
            argtup = (data_freq[ind_frb], ind_frb, y_pred_prob)
            plot_tools.plot_multiple_ranked(argtup, nside=5, \
                fnfigout=options.fnout)
        else:
            plot_tools.plot_multiple_ranked(fnout_ranked, nside=5, \
                                            fnfigout=options.fnout)








            
