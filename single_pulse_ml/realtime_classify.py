import os
import time

import glob

import classify

class RealtimeClassifier():

    def __init__(self, fn_model_freq=None, 
                 fn_model_dm=None, 
                 fn_model_time=None,
                 fn_model_mb=None,
                 ):

        if fn_model_freq is not None:    
            self.model_freq = classify.frbkeras.load_model(fn_model_freq)

        if fn_model_time is not None:
            self.model_time = classify.frbkeras.load_model(fn_model_time)

        if fn_model_dm is not None:
            self.model_dm = classify.frbkeras.load_model(fn_model_dm)

        if fn_model_mb is not None:
            self.model_mb = classify.frbkeras.load_model(fn_model_mb)

        self.data_freq = None


    def prep_data(self, fn_data):
        data_freq, y, data_dm, data_mb, params = classify.reader.read_hdf5(fn_data)

        NFREQ = data_freq.shape[1]  
        NTIME = data_freq.shape[2]
        WIDTH = options.twindow

        # low time index, high time index                                                                                                                                
        tl, th = NTIME//2-WIDTH//2, NTIME//2+WIDTH//2

        if data_freq.shape[-1] > (th-tl):
            data_freq = data_freq[..., tl:th]

        self.data_freq = data_freq

if __name__=='__main__':

    dir = '/data2/output/20190118/2019-01-18-01:29:59.R2/triggers/data/data_full*'
    old_files = []

    while True:
        flist = glob.glob(dir)
        print(flist)

        for fn in flist:
            if fn in old_files:
                print("Sleeping for 1 sec")
                time.sleep(1.0)
                continue
            else:
                classify.main()
                old_files.append(fn)
                time.sleep(1.0)