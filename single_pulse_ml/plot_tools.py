import numpy as np

try:
    import matplotlib 
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    from matplotlib import gridspec
except:
    pass

import keras.backend as backend 


def plot_simulated_events(data, labels, figname,
                          NSIDE, NFREQ, NTIME, cmap='RdBu'):
    """ Make series of waterfall plots of training / test 
    set. 
    """

    NFIG=NSIDE**2
    lab_dict = {0 : 'RFI', 1 : 'FRB'}

    fig = plt.figure(figsize=(15,15))
    for ii in range(NFIG):
        plt.subplot(NSIDE,NSIDE,ii+1)
        plt.imshow(data[ii].reshape(-1, NTIME), 
                   aspect='auto', interpolation='nearest', 
                   cmap=cmap, vmin=-3, vmax=3)
        plt.axis('off')
        plt.colorbar()
        plt.title(lab_dict[labels[ii]])
        plt.xlim(125-32,125+32)
    
    fig.savefig('%s_rfi.png' % figname)

    fig = plt.figure(figsize=(15,15))
    for ii in range(NFIG):
        plt.subplot(NSIDE,NSIDE,ii+1)
        plt.imshow(data[-ii-1].reshape(-1, NTIME), 
                   aspect='auto', interpolation='nearest', 
                   cmap=cmap, vmin=-3, vmax=3)
        plt.axis('off')
        plt.colorbar()
        plt.title(lab_dict[labels[ii]])
        plt.xlim(125-32,125+32)

    fig.savefig('%s_frb.png' % figname)

def plot_gallery(data_arr, titles, h, w, n_row=3, n_col=4, 
                    figname=None, cmap='RdBu', suptitle=''):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.suptitle(suptitle, fontsize=35, color='blue', alpha=0.5)
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(min(n_row * n_col, len(data_arr))):
        d_arr = data_arr[i].reshape((h, w))
        d_arr -= np.median(d_arr)
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(d_arr, cmap=cmap, aspect='auto')        
        plt.title(titles[i], size=12, color='red')
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

def plot_ranked_triggers(data, prob_arr, h=None, w=None, ascending=False, outname='out'):

    assert len(data.shape) == 3, "data should be (batchsize, nside, nside)"

    ranking = np.argsort(prob_arr[:, 0])

    if ascending == True:
        ranking = ranking[::-1]
        title_str = 'RFI most probable'
        outname = outname + 'rfi.png'
    elif ascending == 'mid':
        cp = np.argsort(abs(prob_arr[:,0]-0.5))
        ranking = cp[:h*w]
        title_str = 'Marginal events'
        outname = outname + 'marginal.png'
        print(prob_arr[ranking,0])
    else:
        title_str = 'FRB most probable'
        outname = outname + 'FRB.png '

    fig = plt.figure(figsize=(15,15))

    if h is None:
        h = data.shape[1]
    if w is None:
        w = data.shape[2]

    for ii in range(min(h*w, len(prob_arr))):
        plt.subplot(h, w, ii+1)
        plt.imshow(data[ranking[ii]], 
            cmap='RdBu', interpolation='nearest', 
            aspect='auto')#, vmin=-2, vmax=4)
        plt.axis('off')
#        plt.title(11, 25, str(round(prob_arr[ranking[ii], 1], 3)), fontsize=14)
        plt.title(str(round(prob_arr[ranking[ii], 1], 2)), fontsize=14)

    plt.suptitle(title_str, fontsize=40)

    if outname is not None:
        fig.savefig(outname)

    plt.show()

def plot_image_probabilities(FT_arr, DT_arr, FT_prob_spec, DT_prob_spec):

    assert (len(FT_arr.shape)==2) and (len(DT_arr.shape)==2), "Input data should be (nfreq, ntimes)"

    gs2 = gridspec.GridSpec(4, 3)
    ax1 = plt.subplot(gs2[:2, :2])
    ax1.xaxis.set_ticklabels('')
    ax1.yaxis.set_ticklabels('')
    plt.ylabel('Freq', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    ax1.imshow(FT_arr, cmap='RdBu', interpolation='nearest', aspect='auto')

    ax2 = plt.subplot(gs2[:2, 2:])
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    plt.ylabel('probability', fontsize=18)
    ax2.bar([0, 1], FT_prob_spec, color='red', alpha=0.75)
    plt.xticks([0.5, 1.5], ['RFI', 'Pulse'])
    plt.ylim(0, 1)
    plt.xlim(-.25, 2.)

    ax3 = plt.subplot(gs2[2:, :2])
    ax3.xaxis.set_ticklabels('')
    ax3.yaxis.set_ticklabels('')
    plt.ylabel('Freq', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    ax3.imshow(DT_arr, cmap='RdBu', interpolation='nearest', aspect='auto')

    ax4 = plt.subplot(gs2[2:, 2:])
    ax4.yaxis.set_label_position('right')
    ax4.yaxis.tick_right()
    plt.ylabel('probability', fontsize=18)
    ax4.bar([0, 1], DT_prob_spec, color='red', alpha=0.75)
    plt.xticks([0.5, 1.5], ['RFI', 'Pulse'])
    plt.ylim(0, 1)
    plt.xlim(-.25, 2.)

    plt.suptitle('TensorFlow Deep Learn', fontsize=45, )


#
# Code scraped from https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py
# for layer visualization 
#


class VisualizeLayers:
    """ Class to visualize the hidden 
    layers of a deep neural network in 
    keras. 
    """

    def __init__(self, model):
        self._model = model 
        self._NFREQ = model.get_input_shape_at(0)[1]
        self._NTIME = model.get_input_shape_at(0)[2]
        self.grid_counter = 0

    def print_layers(self):
        for layer in self._model.layers:
            print("%s: %10s" % (layer.name, layer.input.shape))

    def imshow_custom(self, data, **kwargs):
        plt.imshow(data, aspect='auto', interpolation='nearest', 
                         **kwargs)

    def get_activations(self, model_inputs, 
                        print_shape_only=True, 
                        layer_name=None):

        print('----- activations -----')
        activations = []
        inp = self._model.input

        model_multi_inputs_cond = True
        if not isinstance(inp, list):
            # only one input! let's wrap it in a list.
            inp = [inp]
            model_multi_inputs_cond = False

        outputs = [layer.output for layer in self._model.layers if
                   layer.name == layer_name or layer_name is None]  # all layer outputs

        funcs = [backend.function(inp + [backend.learning_phase()], [out]) for out in outputs]  # evaluation functions

        if model_multi_inputs_cond:
            list_inputs = []
            list_inputs.extend(model_inputs)
            list_inputs.append(0.)
        else:
            list_inputs = [model_inputs, 0.]

        # Learning phase. 0 = Test mode (no dropout or batch normalization)
        # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
        layer_outputs = [func(list_inputs)[0] for func in funcs]

        # Append input data
        activations.append(model_inputs)

        for layer_activations in layer_outputs:
            activations.append(layer_activations)

        return activations

    def im_feature_layer(self, activation, cmap='viridis'):
        N_SUBFIG = activation.shape[-1]

        if N_SUBFIG==1:
            #figsize = (7,7)
            cmap = 'RdBu'
            ax = plt.subplot2grid((16,16), (self.grid_counter, 6), colspan=4, rowspan=4)            
            self.grid_counter += 5 # Add one extra unit of space 
            self.imshow_custom(activation[0, :, :, 0], cmap=cmap)
            plt.axis('off')
            return
        else:
            figwidth = 4*N_SUBFIG*activation.shape[1]//self._NFREQ
            figheight = figwidth//N_SUBFIG
            figsize = (figwidth, figheight)
            # ax = plt.subplot2grid((16,16), (self.grid_counter, 8-N_SUBFIG//2), 
            #                       colspan=N_SUBFIG, rowspan=activation.shape[1]//self._NFREQ)
            # self.grid_counter += 4#activation.shape[1]//self._NFREQ

#        fig = plt.figure(figsize=figsize)

        for ii in range(N_SUBFIG):
#            ax = fig.add_subplot(1, N_SUBFIG, ii+1)
            size=1+int(np.round(activation.shape[1]/self._NFREQ))
            start_grid = 8 - N_SUBFIG*size//2
            print(self.grid_counter, start_grid + ii*size, self.grid_counter+size, start_grid + ii*size+size)
            ax = plt.subplot2grid((16,16), (self.grid_counter, start_grid + ii*size), 
                        colspan=size, rowspan=size)
#            self.grid_counter += size
            self.imshow_custom(activation[0, :, :, ii], cmap=cmap)
            plt.axis('off')

        self.grid_counter += size

        #plt.show()

    def im_all(self, activations):
        fig = figure(figsize=(15,15))
        for activation in activations:
            print(activation.shape)
            if activation.shape[-1]==2: # For binary classification
                activation = activation[0]
                activation[0] = 0.25 # Hack for now, visualizing.
                ind = np.array([0, 1])
                width = 0.75
#                fig, ax = plt.subplots()
                ax = plt.subplot2grid((16,16), (13, 6), colspan=4, rowspan=3)

                rects1 = ax.bar(ind[1], activation[1], width, color='r', alpha=0.5)
                rects2 = ax.bar(ind[0], activation[0], width, color='green', alpha=0.5)

                ax.set_xticks(ind + width / 2)
                ax.set_xticklabels(('Noise', 'FRB'))
                ax.set_ylim(0, 1.25)
                ax.set_xlim(-0.25, 2.0)
            elif activation.shape[-1]>64:
                ax = plt.subplot2grid((16,16), (12, 0), colspan=16, rowspan=1)

                activation = activation*np.ones([8, 1])
                plt.imshow((activation[:, :]), interpolation='nearest', 
                           cmap='Greys')
                #plt.show()
            else:
                self.im_feature_layer(activation)



