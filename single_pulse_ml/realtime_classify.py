import os
import time

import optparse
import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib as mpl
mpl.use('pdf', warn=False)

import classify

class RealtimeClassifier():

    def __init__(self, fn_model_freq=None, 
                 fn_model_dm=None, 
                 fn_model_time=None,
                 fn_model_mb=None,
                 twindow=64, nfreq=32, ntime=64):

        if fn_model_freq is not None:    
            self.model_freq = classify.frbkeras.load_model(fn_model_freq)

        if fn_model_time is not None:
            self.model_time = classify.frbkeras.load_model(fn_model_time)

        if fn_model_dm is not None:
            self.model_dm = classify.frbkeras.load_model(fn_model_dm)

        if fn_model_mb is not None:
            self.model_mb = classify.frbkeras.load_model(fn_model_mb)

        self.data_freq = None
        self.twindow = twindow
        self.NFREQ = nfreq
        self.NTIME = ntime

    def prep_data(self, fn_data):
        data_freq, y, data_dm, data_mb, params = classify.reader.read_hdf5(fn_data)

        self.NFREQ = data_freq.shape[1]  
        self.NTIME = data_freq.shape[2]
        WIDTH = self.twindow

        # low time index, high time index                                                                                                                                
        tl, th = self.NTIME//2-WIDTH//2, self.NTIME//2+WIDTH//2

        if data_freq.shape[-1] > (th-tl):
            data_freq = data_freq[..., tl:th]

        self.data_freq = data_freq

if __name__=='__main__':

    parser = optparse.OptionParser(prog="classify.py", \
                        version="", \
                        usage="%prog DATADIR FN_MODEL [OPTIONS]", \
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
                       default='ranked')

    parser.add_option('--nside', dest='nside', type='int', \
                       help="number of rows/cols of subplots per figure", \
                       default=7)

    options, args = parser.parse_args()

    assert len(args)==2, "Arguments are FN_DATA FN_MODEL [OPTIONS]"

    data_dir = args[0]
    fn_model_freq = args[1]

    print("Using data directory %s" % data_dir)
    print("Using keras model in %s" % fn_model_freq)

    SLEEPTIME = 0.1 # seconds 
    old_files = []
    RT_Classifier = RealtimeClassifier(fn_model_freq=fn_model_freq,
                                       fn_model_dm=options.fn_model_dm, 
                                       fn_model_time=options.fn_model_time,
                                       fn_model_mb=options.fn_model_mb,
                                       twindow=64, nfreq=32, ntime=64)

    while True:
        flist = glob.glob(data_dir + '*.hdf5')
        for fn in flist:
            if fn in old_files:
                time.sleep(SLEEPTIME)
                continue
            else:
                try:
                    RT_Classifier.prep_data(fn)
                    data = RT_Classifier.data_freq
                    model = RT_Classifier.model_freq

                    classify.classify(data, model)

                    old_files.append(fn)
                    time.sleep(SLEEPTIME)
                except:
                    print("Error classifying: %s" % fn)
                    old_files.append(fn)
                    time.sleep(SLEEPTIME)                    
                    
