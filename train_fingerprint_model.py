"""
Specifies, trains and evaluates (cross-validation) neural fingerprint model.

Performs early-stopping on the validation set.

The model is specified inside the main() function, which is a demonstration of this code-base

"""
from __future__ import print_function

import time
import numpy as np
import sklearn.metrics as metrics

import warnings

import keras.backend as backend

import KerasNeuralfingerprint.utils as utils
import KerasNeuralfingerprint.data_preprocessing as data_preprocessing
import KerasNeuralfingerprint.fingerprint_model_matrix_based as fingerprint_model_matrix_based
import KerasNeuralfingerprint.fingerprint_model_index_based as fingerprint_model_index_based

from matplotlib import pyplot


#np.set_printoptions(precision=5)

def lim(float, precision = 5):
    return ("{0:."+str(precision)+"f}").format(float)



def save_model_visualization(model, filename='model.png'):
    '''
    Requires the 'graphviz' software package
    '''
    try:
        from keras.utils.visualize_util import plot
        plot(model, filename, show_shapes=1)
    except:
        import traceback
        print('\nsave_model_visualization() failed with exception:',traceback.format_exc())



def predict(data, model):
    '''
    Returns a tensor containing the DNN's predictions for the given list of batches <data>
    '''
    pred = []    
    for batch in data:
        if len(batch)==2:
            batch = batch[0]
        pred.append(model.predict_on_batch(batch))
    return np.concatenate(pred)



def eval_metrics_on(predictions, labels):
    '''
    assuming this is a regression task; labels are continuous-valued floats
    
    returns most regression-related scores for the given predictions/targets as a dictionary:
    
        r2, mean_abs_error, mse, rmse, median_absolute_error, explained_variance_score
    '''
    if len(labels[0])==2: #labels is list of data/labels pairs
        labels = np.concatenate([l[1] for l in labels])
    predictions = predictions[:,0]
    
    r2                       = metrics.r2_score(labels, predictions)
    mean_abs_error           = np.abs(predictions - labels).mean()
    mse                      = ((predictions - labels)**2).mean()
    rmse                     = np.sqrt(mse)
    median_absolute_error    = metrics.median_absolute_error(labels, predictions) # robust to outliers
    explained_variance_score = metrics.explained_variance_score(labels, predictions) # best score = 1, lower is worse
    return {'r2':r2, 'mean_abs_error':mean_abs_error, 'mse':mse, 'rmse':rmse, 
            'median_absolute_error':median_absolute_error, 
            'explained_variance_score':explained_variance_score}



def test_on(data, model, description='test_data score:'):
    '''
    Returns the model's mse on the given data
    '''
    scores=[]
    weights =[]
    for v in data:
        weights.append(v[1].shape) # size of batch
        scores.append( model.test_on_batch(x=v[0], y=v[1]))
    weights = np.array(weights)
    s=np.mean(np.array(scores)* weights/weights.mean())
    if len(description):
        print(description, lim(s))
    return s



def get_model_params(model):
    weight_values = []
    for lay in model.layers:
        weight_values.extend( backend.batch_get_value(lay.weights))
    return weight_values



def set_model_params(model, weight_values):
    symb_weights = []
    for lay in model.layers:
        symb_weights.extend(lay.weights)
    assert len(symb_weights) == len(weight_values)
    for model_w, w in zip(symb_weights, weight_values):
        backend.set_value(model_w, w)
        
        
        
def save_model_weights(model, filename = 'fingerprint_model_weights.npz'):
    ws = get_model_params(model)
    np.savez(filename, ws)



def load_model_weights(model, filename = 'fingerprint_model_weights.npz'):
    ws = np.load(filename)
    set_model_params(model, ws[ws.keys()[0]])
    
    

def update_lr(model, initial_lr, relative_progress, total_lr_decay):
    """
    exponential decay
    
    initial_lr: any float (most reasonable values are in the range of 1e-5 to 1)
    total_lr_decay: value in (0, 1] -- this is the relative final LR at the end of training
    relative_progress: value in [0, 1] -- current position in training, where 0 == beginning, 1==end of training and a linear interpolation in-between
    """
    assert total_lr_decay > 0 and total_lr_decay <= 1
    model.optimizer.lr.set_value(initial_lr * total_lr_decay**(relative_progress))
    


def train_model(model, train_data, valid_data, test_data, 
                 batchsize = 100, num_epochs = 100, train = True, 
                 initial_lr=3e-3, total_lr_decay=0.2, verbose = 1):
    """
    Main training loop for the DNN.
    
    Input:
    ---------    
    
    train_data, valid_data, test_data:
    
        lists of tuples (data-batch, labels-batch)
    
    total_lr_decay:
        
        value in (0, 1] -- this is the inverse total LR reduction factor over the course of training

    verbose:
        
        value in [0,1,2] -- 0 print minimal information (when training ends), 1 shows training loss, 2 shows training and validation loss after each epoch
    
    
    Returns:
    -----------
    
        model (keras model object) -- model/weights selected by early-stopping on the validation set (model at epoch with lowest validation error)
        
        3-tuple of train/validation/test dictionaries (as returned by eval_metrics_on(...) )
        
        2-tuple of training/validation-set MSE after each epoch of training
        
    """

    if train:
        
        log_train_mse = []
        log_validation_mse = []
        
        if verbose>0:
            print('starting training (compiling)...')
        
        best_valid = 9e9
        model_params_at_best_valid=[]
        
        times=[]
        for epoch in range(num_epochs):
            update_lr(model, initial_lr, epoch*1./num_epochs, total_lr_decay)
            batch_order = np.random.permutation(len(train_data))
            losses=[]
            t0 = time.clock()
            for i in batch_order:
                loss = model.train_on_batch(x=train_data[i][0], y=train_data[i][1], check_batch_dim=False)
                losses.append(loss)
            times.append(time.clock()-t0)
            val_mse = test_on(valid_data,model,'valid_data score:' if verbose>1 else '')
            if best_valid > val_mse:
                best_valid = val_mse
                model_params_at_best_valid = get_model_params(model) #kept in RAM (not saved to disk as that is slower)
            if verbose>0:
                print('Epoch',epoch+1,'completed with average loss',lim(np.mean(losses)))
            log_train_mse.append(np.mean(losses))
            log_validation_mse.append(val_mse)
            
        # excludes times[0] as it includes compilation time
        print('Training @',lim(1./np.mean(times[1:])),'epochs/sec (',lim(batchsize*len(train_data)/np.mean(times[1:])),'examples/s)')
    
    
    train_end  = test_on(train_data,model,'train mse (final):     ')
    val_end    = test_on(valid_data,model,'validation mse (final):')
    test_end   = test_on(test_data, model,'test  mse (final):     ')
    
    set_model_params(model, model_params_at_best_valid)
    
    training_data_scores   = eval_metrics_on(predict(train_data,model), train_data)
    validation_data_scores = eval_metrics_on(predict(valid_data,model), valid_data)
    test_data_scores       = eval_metrics_on(predict(test_data,model),  test_data)
    
    
    print('training set mse (best_val):  ', lim(training_data_scores['mse']))
    print('validation set mse (best_val):', lim(validation_data_scores['mse']))
    print('test set mse (best_val):      ', lim(test_data_scores['mse']))
    
    return model, (training_data_scores, validation_data_scores, test_data_scores), (log_train_mse, log_validation_mse)







def plot_training_mse_evolution(data_lists, legend_names=[], ylabel = 'MSE', xlabel = 'training epoch', legend_location='best'):
    
    _colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    figure = pyplot.figure()
    ax  = figure.add_subplot(111)
    ax.set_axisbelow(True)
    if len(legend_names):
        assert len(legend_names)==len(data_lists), 'you did not provide enough or too many labels for the graph'
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    pyplot.grid(b=True, which='major', color='lightgray', linestyle='--')
    if len(legend_names) != len(data_lists):
        legend_names = [' ' for x in data_lists]
    for i, data in enumerate(data_lists):
        assert len(data)==len(data_lists[0])
        pyplot.plot(np.arange(1,len(data)+1), data, 
                    _colors[i%len(_colors)], linestyle='-', marker='o', 
                    markersize=5, markeredgewidth=0.5, linewidth=2.5, label=legend_names[i])
    if len(legend_names[0]):
        ax.legend(loc=legend_location, shadow=0, prop={'size':14}, numpoints=1)
    
    
    
    
    
def crossvalidation_example(use_matrix_based_implementation = False, plot_training_mse = False):
    """
    Demonstration of data preprocessing, network configuration and (cross-validation) Training & testing
    
    There are two different (but equivalent!) implementations of neural-fingerprints, 
    which can be selected with the binary parameter <use_matrix_based_implementation>
    
    """
    # for reproducibility
    np.random.seed(1338)  
    
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~
    num_epochs = 150
    batchsize  = 20   #batch size for training
    L2_reg     = 4e-3
    batch_normalization = 0
    #~~~~~~~~~~~~~~~~~~~~~~~~~
    fp_length = 51  # size of the final constructed fingerprint vector
    conv_width = 50 # number of filters per fingerprint layer
    fp_depth = 3    # number of convolutional fingerprint layers
    #~~~~~~~~~~~~~~~~~~~~~~~~~
    n_hidden_units = 100
    predictor_MLP_layers = [n_hidden_units, n_hidden_units, n_hidden_units]    
    #~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
    # total number of cross-validation splits to perform
    crossval_total_num_splits = 10
    
    
    # select the data that will be loaded or provide different data 
    data, labels = utils.filter_data(utils.load_delaney, data_cache_name='data/delaney')
#    data, labels = utils.filter_data(utils.load_Karthikeyan_MeltingPoints, data_cache_name='data/Karthikeyan_MeltingPoints')
    print('# of valid examples in data set:',len(data))
    

    
    train_mse = []
    val_mse   = []
    test_mse  = []
    test_scores = []
    
    if use_matrix_based_implementation:
        fn_build_model   = fingerprint_model_matrix_based.build_fingerprint_regression_model
    else:
        fn_build_model   = fingerprint_model_index_based.build_fingerprint_regression_model
    
    
    print('Naive baseline (using mean): MSE =', lim(np.mean((labels-labels.mean())**2)), '(RMSE =', lim(np.sqrt(np.mean((labels-labels.mean())**2))),')')
   
    
    
    for crossval_split_index in range(crossval_total_num_splits):
        print('\ncrossvalidation split',crossval_split_index+1,'of',crossval_total_num_splits)
    
        traindata, valdata, testdata = utils.cross_validation_split(data, labels, crossval_split_index=crossval_split_index, 
                                                                    crossval_total_num_splits=crossval_total_num_splits, 
                                                                    validation_data_ratio=0.1)
        
        
        train, valid_data, test_data = data_preprocessing.preprocess_data_set_for_Model(traindata, valdata, testdata, 
                                                                     training_batchsize = batchsize, 
                                                                     testset_batchsize = 1000)
        

        model = fn_build_model(fp_length = fp_length, fp_depth = fp_depth, 
                               conv_width = conv_width, predictor_MLP_layers = predictor_MLP_layers, 
                               L2_reg = L2_reg, num_input_atom_features = 62, 
                               num_bond_features = 6, batch_normalization = batch_normalization)
        

        
        model, (train_scores_at_valbest, val_scores_best, test_scores_at_valbest), train_valid_mse_per_epoch = train_model(model, train, valid_data, test_data, 
                                     batchsize = batchsize, num_epochs = num_epochs, train=1)
        train_mse.append(train_scores_at_valbest['mse'])
        val_mse.append(val_scores_best['mse'])
        test_mse.append(test_scores_at_valbest['mse'])
        test_scores.append(test_scores_at_valbest)
        
        if plot_training_mse:
            plot_training_mse_evolution(train_valid_mse_per_epoch, ['training set MSE (+regularizer)', 'validation set MSE'])
            pyplot.draw()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pyplot.pause(0.0001)
    
        
    
    


    
    print('\n\nCrossvalidation complete!\n')
    
    print('Mean training_data MSE =', lim(np.mean(train_mse)),                   '+-', lim(np.std(train_mse)/np.sqrt(crossval_total_num_splits)))
    print('Mean validation    MSE =', lim(np.mean(val_mse)),                     '+-', lim(np.std(val_mse)/np.sqrt(crossval_total_num_splits)))
    print('Mean test_data     MSE =', lim(np.mean(test_mse)),                    '+-', lim(np.std(test_mse)/np.sqrt(crossval_total_num_splits)))
    print('Mean test_data RMSE    =', lim(np.mean(np.sqrt(np.array(test_mse)))), '+-', lim(np.std(np.sqrt(np.array(test_mse)))/np.sqrt(crossval_total_num_splits)))
    
    
    avg_test_scores = np.array([x.values() for x in test_scores]).mean(0)
    avg_test_scores_dict = dict(zip(test_scores[0].keys(), avg_test_scores))
    print()
    for k,v in avg_test_scores_dict.items():
        if k not in ['mse','rmse']:#filter duplicates
            print('Test-set',k,'=',lim(v))
                
    
    return model #this is the last trained model
    
    
    
    
if __name__=='__main__':
    
    
    plot_training_mse = 1
    
    
    model = crossvalidation_example(1, plot_training_mse = plot_training_mse)
    
    
    # to save the model weights use e.g.:
    save_model_weights(model, 'trained_fingerprint_model.npz')
    
    # to load the saved model weights use e.g.:
    load_model_weights(model, 'trained_fingerprint_model.npz')
    
    
    #this saves an image of the network's computational graph (an abstract form of it)
    # beware that this requires the 'graphviz' software!
    save_model_visualization(model, filename = 'fingerprintmodel.png')


    pyplot.show()
