"""
Specifies, trains and evaluates (cross-validation) neural fingerprint model.

Performs early-stopping on the validation set.

The model is specified inside the main() function, which is a demonstration of this code-base

"""

import time
import numpy as np

import keras.backend as backend

import KerasNeuralfingerprint.utils as utils

import KerasNeuralfingerprint.data_preprocessing__matrix_based as data_preprocessing__matrix_based
import KerasNeuralfingerprint.fingerprint_model_matrix_based as fingerprint_model_matrix_based

#import KerasNeuralfingerprint.data_preprocessing__index_based as data_preprocessing__index_based
#import KerasNeuralfingerprint.fingerprint_model_index_based as fingerprint_model_index_based





def test_on(data, model, description='test_data score:'):
    scores=[]
    weights =[]
    for v in data:
        weights.append(v[1].shape) # size of batch
        scores.append( model.test_on_batch(x=v[0], y=v[1]))
    weights = np.array(weights)
    s=np.mean(np.array(scores)* weights/weights.mean())
    if len(description):
        print(description, s)
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
    
    train_data, valid_data, test_data:
    
        lists of tuples (data-batch, labels-batch)
    
    total_lr_decay:
        
        value in (0, 1] -- this is the inverse total LR reduction factor over the course of training

    verbose:
        
        value in [0,1,2] -- 0 print minimal information (when training ends), 1 shows training loss, 2 shows training and validation loss after each epoch
    
    """

    if train:
        if verbose>0:
            print 'starting training (compiling)...'
        
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
                model_params_at_best_valid = get_model_params(model)
            if verbose>0:
                print 'Epoch',epoch+1,'completed with average loss',np.mean(losses)
            
        # excludes times[0] as it includes compilation time
        print 'Training @',1./np.mean(times[1:]),'epochs/sec (',batchsize*len(train_data)/np.mean(times[1:]),'examples/s)'
    test_on(train_data,model,'train score (final):')
    test_on(valid_data,model,'validation score (final):')
    test_on(test_data, model,'test  score (final):')
    
    set_model_params(model, model_params_at_best_valid)
    
    train_valbest   = test_on(train_data,model,'train score (best_val):')
    val_best        = test_on(valid_data,model,'validation score (best_val):')
    test_at_valbest = test_on(test_data, model,'test  score (best_val):')
    
    return model, train_valbest, val_best, test_at_valbest



    
    
    
    
    
    
def main(use_matrix_based_implementation = False):
    """
    Demo of data preprocessing, network configuration and (cross-validation) Training & testing
    
    There are two different (but equivalent!) implementations of neural-fingerprints, 
    which can be selected with the binary parameter <use_matrix_based_implementation>
    
    """
    # for reproducibility
    np.random.seed(1338)  

    n_hidden_units = 100

    num_epochs = 100
    batchsize = 20
    fp_length = 50
    conv_width = 50
    fp_depth = 3
    predictor_MLP_layers = [n_hidden_units, n_hidden_units, n_hidden_units]
    L2_reg = 4e-3
    batch_normalize = 0
    
    
    # total number of cross-validation splits to perform
    crossval_total_num_splits = 10
    
    
    # select the data that will be loaded or provide different data 
    data, labels = utils.load_delaney()
#    data, labels = utils.filter_data(utils.load_Karthikeyan_MeltingPoints, data_cache_name='data/Karthikeyan_MeltingPoints')
    
    
    
    
#    should obtain (on delaney):
#    Mean validation MSE = 0.286070278287 +- 0.00692366129073
#    Mean test_data MSE = 0.391990280151 +- 0.034257187573
#    Mean test_data RMSE = 0.620833234262 +- 0.0256054201042
#    trains @ 2.6 epochs/sec (MKL-linked numpy, CPU-mode, batchsize of 20)
    

    
    
    train_mse = []
    val_mse   = []
    test_mse  = []
    
    if use_matrix_based_implementation:
        fn_preprocessing = data_preprocessing__matrix_based.preprocess_data_set_for_Model
        fn_build_model   = fingerprint_model_matrix_based.build_fingerprint_regression_model
    else:
        raise NotImplementedError('coming soon')
        #fn_preprocessing = data_preprocessing__index_based.preprocess_data_set_for_Model
        #fn_build_model   = fingerprint_model_index_based.build_fingerprint_regression_model
    
    
    print 'Naive baseline (using mean): MSE =', np.mean((labels-labels.mean())**2), '(RMSE =', np.sqrt(np.mean((labels-labels.mean())**2)),')'
   
    
    
    for crossval_split_index in range(crossval_total_num_splits):
        print '\ncrossvalidation split',crossval_split_index+1,'of',crossval_total_num_splits
    
        traindata, valdata, testdata = utils.cross_validation_split(data, labels, crossval_split_index=crossval_split_index, 
                                                                    crossval_total_num_splits=crossval_total_num_splits, 
                                                                    validation_data_ratio=0.1)
        
        
        train, valid_data, test_data = fn_preprocessing(traindata, valdata, testdata, 
                                                                     training_batchsize = batchsize, 
                                                                     testset_batchsize = 1000)
        

        model = fn_build_model(fp_length = fp_length, fp_depth = fp_depth, 
                                                        conv_width = conv_width, 
                                                 predictor_MLP_layers = predictor_MLP_layers, 
                                                 L2_reg = L2_reg, num_input_atom_features = 62, 
                                                 num_bond_features = 6, batch_normalize = batch_normalize)
        
        model, train_mse_at_valbest, val_mse_best, test_mse_at_valbest = train_model(model, train, valid_data, test_data, 
                                     batchsize = batchsize, num_epochs = num_epochs, train=1)
        train_mse.append(train_mse_at_valbest)
        val_mse.append(val_mse_best)
        test_mse.append(test_mse_at_valbest)

    print
    print
    print 'Crossvalidation complete!'
    print
    print 'Mean training_data MSE =', np.mean(train_mse), '+-', np.std(train_mse)/np.sqrt(crossval_total_num_splits)
    print 'Mean validation    MSE =', np.mean(val_mse), '+-', np.std(val_mse)/np.sqrt(crossval_total_num_splits)
    print 'Mean test_data     MSE =', np.mean(test_mse), '+-', np.std(test_mse)/np.sqrt(crossval_total_num_splits)
    print 'Mean test_data RMSE    =', np.mean(np.sqrt(np.array(test_mse))), '+-', np.std(np.sqrt(np.array(test_mse)))/np.sqrt(crossval_total_num_splits)

    
    
    
    
    
    
if __name__=='__main__':
    
    
    main(1)



