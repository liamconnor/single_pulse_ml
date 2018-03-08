import numpy as np

try:
    import matplotlib 
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    from matplotlib import gridspec
except:
    print("Didn't work")
    pass

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

    fig.savefig(figname)

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

def plot_ranked_trigger(data, prob_arr, h=6, w=6, ascending=False, outname='out'):
    assert len(data.shape) == 3, "data should be (batchsize, nside, nside)"

    ranking = np.argsort(prob_arr[:, 0])

    if ascending == True:
        ranking = ranking[::-1]
        title_str = 'RFI most probable'
        outname = outname + 'rfi.png'
    elif ascending == 'mid':
#        cp = np.argsort(abs(prob_arr[:,0]-0.5))
#        ranking = cp[:h*w]
        inflection = np.argmax(abs(np.diff(prob_arr[:,0][ranking])))
        ranking = ranking[inflection-h*w/2:inflection+h*w/2]
        title_str = 'Marginal events'
        outname = outname + 'marginal.png'
        print(prob_arr[ranking,0])
    else:
        title_str = 'FRB most probable'
        outname = outname + 'FRB.png'

    fig = plt.figure(figsize=(15,15))

    for ii in range(min(h*w, len(prob_arr))):
        plt.subplot(h, w, ii+1)
        plt.imshow(data[ranking[ii]], 
            cmap='Greys', interpolation='nearest', 
            aspect='auto', vmin=-10, vmax=10, 
            extent=[0, 1, 400, 800])
        #plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.title('p='+str(np.round(prob_arr[ranking[ii], 0], 5)), fontsize=12)

        if ii % w == 0:
            plt.ylabel("Freq", fontsize=14)
        if ii >= (h*w-w):
            plt.xlabel("Time", fontsize=14)

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


class VisualizeLayers:
    """ Class to visualize the hidden 
    layers of a deep neural network in 
    keras.
    """
    import keras.backend as backend 

    def __init__(self, model):
        self._model = model 
        self._NFREQ = model.get_input_shape_at(0)[1]
        self._NTIME = model.get_input_shape_at(0)[2]
        self.grid_counter = 0
        # Create empty list for non-redundant activations
        self._activations_nonred = []
        self._NFREQ_min = min([mm.input.shape[1] for mm in model.layers])

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

    def plot_feature_layer(self, activation, NSIDE=16):
        N_SUBFIG = activation.shape[-1]

        if N_SUBFIG==1:

            ax = plt.subplot2grid((NSIDE,NSIDE), 
                          (self.grid_counter, 3*NSIDE//8), 
                          colspan=NSIDE//4, rowspan=NSIDE//4) 
            plt.plot(activation[0,:,0])
            return 

        for ii in range(N_SUBFIG):
            size=int(activation.shape[1] / self._NFREQ_min)
#            size=int(np.round(4*activation.shape[1]/self._NFREQ * NSIDE//32))
#            size=min(size, NSIDE//8)
            start_grid = NSIDE//2 - N_SUBFIG*size//2
            print(NSIDE, self.grid_counter, start_grid + ii*size, size)
            ax = plt.subplot2grid((NSIDE,NSIDE), 
                        (self.grid_counter, start_grid + ii*size), 
                        colspan=size, rowspan=size)
            plt.plot(activation[0,:,ii])
            plt.axis('off')

    def im_feature_layer(self, activation, cmap='Greys', NSIDE=16, 
                         start_grid=0, N_SUBFIG=None, skip=1):
        N_SUBFIG = activation.shape[-1] if N_SUBFIG is None else N_SUBFIG

        if N_SUBFIG==1:
#            cmap = 'RdBu'

            ax = plt.subplot2grid((NSIDE,NSIDE), 
                          (self.grid_counter, 3*NSIDE//8), 
                          colspan=NSIDE//4, rowspan=NSIDE//4) 

            print(self.grid_counter,'0')
            self.grid_counter += (NSIDE//4+NSIDE//16) # Add one extra unit of space 
            print(activation.shape)
            data = activation[0,:,:,0]
            data -= np.median(data)
            vmax = 6*np.std(data)
            vmin = -1*np.std(data)
            self.imshow_custom(data,
                                cmap=cmap, extent=[0, 1, 400, 800], vmax=vmax, vmin=vmin)
            print(self.grid_counter,'1')

            plt.xlabel('Time')
            plt.ylabel('Freq [MHz]')

            return

#         for ii in range(N_SUBFIG):
#             size=int(np.round(4*activation.shape[1]/self._NFREQ * NSIDE//32))
#             size=min(size, NSIDE//8)

#             print(size, skip, ii*size*skip)
#             ax = plt.subplot2grid((NSIDE,NSIDE), 
#                         (self.grid_counter, start_grid + ii*size*skip), 
#                         colspan=size, rowspan=size)
#             data = activation[0, :, :, ii]
# #            data -= np.median(data)
#             vmax = 4*np.std(data)
#             vmin = -4*np.std(data)
#             self.imshow_custom(data, cmap=cmap)#, vmax=vmax, vmin=vmin)
#             plt.axis('off')

#        self.grid_counter += (NSIDE//32+int(size))

        #plt.show()

    def im_layers(self, activations, loc_obj, cmap='Greys'):
        
        sizes = loc_obj[0]
        loc = loc_obj[1]

        for jj, activation in enumerate(activations):
            for ii in range(activation.shape[-1]):
                ax = plt.subplot2grid((NSIDE,NSIDE),(self.grid_counter, loc[jj][ii]), 
                                      colspan=sizes[jj], rowspan=sizes[jj]) 

                self.imshow_custom(activation[0,:,:,ii], cmap='Greys')
                plt.axis('off')

            self.grid_counter += (NSIDE//32+int(sizes[jj]))

        plt.show()

    def get_image_index(self, NSIDE=100):
        offset = 0
        sizes = np.array([8, 4, 4, 2])
        N_SUBFIG = np.array([8, 8, 16, 16])
        offset = NSIDE//2 - N_SUBFIG*sizes//2
        loc1 = (offset[0] + np.arange(8)*sizes[0]).astype(int)
        loc2 = (loc1 + (sizes[0]/2 - sizes[1]/2)).astype(int)
        loc3 = (offset[2] + arange(16)*(1+sizes[2])).astype(int)
        offset3 = NSIDE//2 - (loc3[0] + (loc3[-1] - loc3[0])/2.)
        loc3 += int(offset3)
        loc4 = (loc3 + (sizes[2]/2 - sizes[3]/2)).astype(int)
        loc = [loc1, loc2, loc3, loc4]

        loc_obj = (sizes, loc)

        return loc_obj 

    def im_all(self, activations, NSIDE=32, figname=None, color='linen'):
        fig = figure(figsize=(15,15))
        self.grid_counter = 0
        start_grid_map = np.zeros([len(activations)]).astype(int)
        n_neuron_map = [activation.shape[-1] for activation in activations]
        loc_obj = self.get_image_index()

        for kk, activation in enumerate(activations[:]): 
            print(self.grid_counter, kk, activation.shape)
            if kk==0:
                self.im_layers(activation, loc_obj, cmap='Greys')
            elif activation.shape[-1]==2: # For binary classification
                activation = activation[0]
                activation[0] = 0.025 # Hack for now, visualizing.
                ind = np.array([0, 1])
                width = 0.75
                ax = plt.subplot2grid((NSIDE,NSIDE), 
                                      (self.grid_counter, 3*NSIDE//8), 
                                       colspan=NSIDE//4, rowspan=NSIDE//4)

                rects1 = ax.bar(ind[1], activation[1], width, color='r', alpha=0.5)
                rects2 = ax.bar(ind[0], activation[0], width, color='green', alpha=0.5)

                ax.set_xticks(ind + width / 2)
                ax.set_xticklabels(('Noise', 'FRB'))
                ax.set_ylim(0, 1.25)
                ax.set_xlim(-0.25, 2.0)

            elif kk==1:
                self.im_layers(activations[1:5], loc_obj, cmap='Greys')

        # for kk, activation in enumerate(activations[:]): #hack
        #     print(activation.shape)
        #     if activation.shape[-1]==2: # For binary classification
        #         activation = activation[0]
        #         activation[0] = 0.025 # Hack for now, visualizing.
        #         ind = np.array([0, 1])
        #         width = 0.75
        #         ax = plt.subplot2grid((NSIDE,NSIDE), 
        #                               (self.grid_counter, 3*NSIDE//8), 
        #                                colspan=NSIDE//4, rowspan=NSIDE//4)

        #         rects1 = ax.bar(ind[1], activation[1], width, color='r', alpha=0.5)
        #         rects2 = ax.bar(ind[0], activation[0], width, color='green', alpha=0.5)

        #         ax.set_xticks(ind + width / 2)
        #         ax.set_xticklabels(('Noise', 'FRB'))
        #         ax.set_ylim(0, 1.25)
        #         ax.set_xlim(-0.25, 2.0)
        #     elif activation.shape[-1]==1:
        #         self.im_feature_layer(activation, NSIDE=NSIDE, 
        #                                 N_SUBFIG=1, skip=1)
        #     elif (activation.shape[-1]>64) and (activation.shape[-1]<1024):
        #         ax = plt.subplot2grid((NSIDE,NSIDE), (self.grid_counter, 0), 
        #                               colspan=NSIDE, rowspan=NSIDE//32)
        #         activation = activation*np.ones([2, 1])
        #         plt.imshow((activation[:, :]), interpolation='nearest', 
        #                    cmap='RdBu')
        #         plt.axis('off')
        #         self.grid_counter += NSIDE//8
        #         ax.get_yaxis().set_visible(False)

            # elif activation.shape[-1]<64:
            #     N_SUBFIG = n_neuron_map[kk]
            #     if kk==1:
            #         skip = 1
            #         size = int(np.round(4*activation.shape[1]/self._NFREQ * NSIDE//32))
            #         size = min(size, NSIDE//8)       
            #         start_grid = NSIDE//2 - N_SUBFIG*size//2
            #     elif N_SUBFIG==n_neuron_map[kk-1]:
            #         start_grid = start_grid + size//3
            #         skip = size//2                    
            #         size = int(np.round(4*activation.shape[1]/self._NFREQ * NSIDE//32))
            #         size = min(size, NSIDE//8)
            #     else:
            #         start_grid=0
            #         skip = 1

            #     self.im_feature_layer(activation, NSIDE=NSIDE, 
            #                             start_grid=start_grid, N_SUBFIG=N_SUBFIG, skip=skip)

            if figname is not None:
                plt.savefig(figname)#, facecolor=color)

    def make_figure(self, data, NSIDE=32, figname=None):
        dsh = data.shape

        if len(dsh)==2:
            data = data[None,:,:,None]
        elif len(dsh)==3:
            if dsh[0]==1:
                data = data[..., None]
            elif dsh[-1]==1:
                data = data[None]

        # Make sure there's no activation 
        # which has more filters than NSIDE
        for activation in activations:
            if len(activation.shape) > 2:
                NSIDE = max(NSIDE, activation.shape[-1])

        print("Using NSIDE: %d" % NSIDE) 

        self.remove_doubles(activations)
        self.im_all(self._activations_nonred, NSIDE=NSIDE, figname=figname)

