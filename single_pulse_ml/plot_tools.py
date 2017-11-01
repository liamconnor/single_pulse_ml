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
        # Create empty list for non-redundant activations
        self._activations_nonred = []

    def print_layers(self):
        """ Print layer names and shapes of keras model
        """
        for layer in self._model.layers:
            print("%s: %10s" % (layer.name, layer.input.shape))

    def imshow_custom(self, data, **kwargs):
        """ matplotlib imshow with custom arguments
        """
        plt.imshow(data, aspect='auto', interpolation='nearest', 
                         **kwargs)

    def remove_doubles(self, activations):
        """ Remove layers with identical shapes, e.g. 
        dropout layers
        """
        self._activations_nonred.append(activations[0])

        # Start from first element, skip input data
        for ii, activation in enumerate(activations[1:]):
            act_shape = activation.shape
            if act_shape != activations[ii].shape:
                self._activations_nonred.append(activation)

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

    def im_feature_layer(self, activation, cmap='viridis', NSIDE=16):
        N_SUBFIG = activation.shape[-1]

        if N_SUBFIG==1:
            cmap = 'RdBu'
            ax = plt.subplot2grid((NSIDE,NSIDE), 
                          (self.grid_counter, 3*NSIDE//8), 
                          colspan=NSIDE//4, rowspan=NSIDE//4)            
            self.grid_counter += (NSIDE//4+NSIDE//16) # Add one extra unit of space 
            self.imshow_custom(activation[0, :, :, 0], cmap=cmap, extent=[0, 1, 400, 800])
            plt.xlabel('Time')
            plt.ylabel('Freq [MHz]')
            return
        else:
            figwidth = 4*N_SUBFIG*activation.shape[1]//self._NFREQ
            figheight = figwidth//N_SUBFIG
            figsize = (figwidth, figheight)

        for ii in range(N_SUBFIG):
            size=int(np.round(4*activation.shape[1]/self._NFREQ * NSIDE//32))
            size=min(size, NSIDE//8)
            start_grid = NSIDE//2 - N_SUBFIG*size//2

            ax = plt.subplot2grid((NSIDE,NSIDE), 
                        (self.grid_counter, start_grid + ii*size), 
                        colspan=size, rowspan=size)

            self.imshow_custom(activation[0, :, :, ii], cmap=cmap)
            plt.axis('off')

        self.grid_counter += (NSIDE//32+size)

        #plt.show()

    def im_all(self, activations, NSIDE=32, figname=None, color='linen'):
        fig = figure(figsize=(15,15))
        self.grid_counter = 0

        for activation in activations[:]: #hack
            print(activation.shape)
            if activation.shape[-1]==2: # For binary classification
                activation = activation[0]
                activation[0] = 0.025 # Hack for now, visualizing.
                ind = np.array([0, 1])
                width = 0.75
#                fig, ax = plt.subplots()
                ax = plt.subplot2grid((NSIDE,NSIDE), 
                                      (self.grid_counter, 3*NSIDE//8), 
                                       colspan=NSIDE//4, rowspan=NSIDE//4)

                rects1 = ax.bar(ind[1], activation[1], width, color='r', alpha=0.5)
                rects2 = ax.bar(ind[0], activation[0], width, color='green', alpha=0.5)

                ax.set_xticks(ind + width / 2)
                ax.set_xticklabels(('Noise', 'FRB'))
                ax.set_ylim(0, 1.25)
                ax.set_xlim(-0.25, 2.0)
            elif (activation.shape[-1]>64) and (activation.shape[-1]<1024):
                ax = plt.subplot2grid((NSIDE,NSIDE), (self.grid_counter, 0), 
                                      colspan=NSIDE, rowspan=NSIDE//32)
                activation = activation*np.ones([NSIDE//8, 1])
                plt.imshow((activation[:, :]), interpolation='nearest', 
                           cmap='Greys')
                self.grid_counter += NSIDE//8
                ax.get_yaxis().set_visible(False)

            elif activation.shape[-1]<64:
                self.im_feature_layer(activation, NSIDE=NSIDE)

            if figname is not None:
                plt.savefig(figname, facecolor=color)

    def make_figure(self, data, NSIDE=16, figname=None):
        dsh = data.shape

        if len(dsh)==2:
            data = data[None,:,:,None]
        elif len(dsh)==3:
            if dsh[0]==1:
                data = data[..., None]
            elif dsh[-1]==1:
                data = data[None]

        activations = self.get_activations(data)
        self.remove_doubles(activations)
        self.im_all(self._activations_nonred, NSIDE=NSIDE, figname=figname)

