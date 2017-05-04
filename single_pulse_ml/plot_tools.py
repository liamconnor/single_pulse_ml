import numpy as np
import matplotlib 
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def plot_gallery(data_arr, titles, h, w, n_row=3, n_col=4, figname=None, cmap='RdBu'):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(min(n_row * n_col, len(data_arr))):
        d_arr = data_arr[i].reshape((h, w))
        d_arr -= np.median(d_arr)
        plt.subplot(n_row, n_col, i + 1)
#        plt.imshow(data_arr[i].reshape((h, w)), cmap=plt.cm.gray, aspect='auto')
        plt.imshow(d_arr, cmap=cmap, aspect='auto')        
        plt.title(titles[i], size=14, color='red')
        plt.xticks(())
        plt.yticks(())
    if figname:
    	plt.savefig(figname)


def get_title(y, target_names):
    prediction_titles = y.astype(str)
    prediction_titles[prediction_titles=='0'] = target_names[0]
    prediction_titles[prediction_titles=='1'] = target_names[1]

    return prediction_titles

def get_title2(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]]
    true_name = target_names[y_test[i]]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)
