import os
import time

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

    dir = '/data2/output/20190118/2019-01-18-01:29:59.R2/triggers/data/data_full*'
    dir = '../../../arts-analysis/arts-analysis/heimearly10-500/data/data_full*'
    mod = './model/october1_heimdall_crab_simfreq_time.hdf5'
    old_files = []
    RT_Classifier = RealtimeClassifier(mod)

    while True:
        flist = glob.glob(dir)
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
                    
