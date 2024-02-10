# # Noninterpretable Models: XGBoost, Fully Connected NeuralNetworks baselines
# This notebook runs baseline nonlinear noninterpretable models for comparison with state-of-the-art black-box methods.

from __future__ import division
import sys
print(sys.version, sys.platform, sys.executable) #Displays what environment you are actually using.
from contextlib import redirect_stdout

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import RandomizedSearchCV, KFold, PredefinedSplit

from tqdm.keras import TqdmCallback
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TerminateOnNaN
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, SparkTrials
import tensorflow.keras.backend as K

import time
from copy import deepcopy
from tqdm import notebook
from ipywidgets import *

import load_nn_model
import data_utils
import utils
import pathlib
import argparse

parser = argparse.ArgumentParser(description='Fully connected Feed Forward Neural Networks on synthetic data.')

# Data Arguments
parser.add_argument('--load_directory', dest='load_directory',  type=str, default='/home/shibal/Census-Data')
parser.add_argument('--seed', dest='seed',  type=int, default=8)

# Logging Arguments
parser.add_argument('--version', dest='version',  type=int, default=1)
args = parser.parse_args()


# # Import Processed Data
#
# Load directory needs to be updated with the path of the dropbox folder downloaded from the following link: 
# https://www.dropbox.com/sh/piwdz9sbmxjx03f/AACAKqjunrEhaRY9SORRE-Yba?dl=0

load_directory=args.load_directory
version=args.version
seed = args.seed
save_directory = os.path.join(str(pathlib.Path(__file__).parent.absolute()).split('baselines')[0], "results", "BlackBox", "v{}".format(version), "seed{}".format(seed), "fcnn") 

df_X, df_y, _ = data_utils.load_data(load_directory=load_directory,
                                  filename='pdb2019trv3_us.csv',
                                  remove_margin_of_error_variables=True)
np.random.seed(seed)
X_train, y_train, X_val, y_val, X_test, y_test, _, y_scaler = data_utils.process_data(
    df_X,
    df_y,
    val_ratio=0.1, 
    test_ratio=0.1,
    seed=seed,
    standardize_response=False)


X = np.append(X_train, X_val, axis=0)
y = np.append(y_train, y_val, axis=0)
val_fold = np.append(-1*np.ones(y_train.shape), np.zeros(y_val.shape))
ps = PredefinedSplit(test_fold=val_fold)
neg_scorer = make_scorer(mean_squared_error, greater_is_better=False)


ntrials = 1000
pen = np.array(['FC-NN'])
M = pen.shape[0]
df = pd.DataFrame(data={'': pen,
                      'Training MSE': np.zeros(M), 
                      'Validation MSE': np.zeros(M),
                      'Test MSE': np.zeros(M),
                      'Test RMSE': np.zeros(M),
                      'Nonzeros': np.zeros(M)})
df = df.set_index('')



# # Fully Connected Neural Networks
#
# Uses hyperopt library to tune the tensorflow models

save_path = os.path.join(save_directory, "FC-NN")
os.makedirs(save_path, exist_ok=True)

# Parameter space
space = {'units': hp.choice('units', [64, 128, 256, 512]),
         'layers_dense': hp.choice('layers_dense', [2, 3, 4, 5, 6]),
         'dropout': hp.choice('dropout', [0.1,0.2,0.3]),
         'batch_size': hp.choice('batch_size', [64, 128]),
         'epochs':  hp.choice('epochs', [25, 50, 75, 100]),
         'optimizer': hp.choice('optimizer',[tf.keras.optimizers.Adam]),
         'learning_rate': hp.choice('learning_rate', [1e-1, 1e-2, 1e-3, 1e-4]),
        }


def f_model(params):
    
    model = load_nn_model.create_model(
        input_shape=(X_train.shape[1],),
        layers_dense=params['layers_dense'],
        units=params['units'], 
        dropout=params['dropout'],
    )
                              
    # Compile model
    model.compile(loss='mse',
                  optimizer=params['optimizer'](learning_rate=params['learning_rate']),
                  metrics=['mean_squared_error'])
    
    # Train model
    callbacks = [TerminateOnNaN(), TqdmCallback(verbose=0)]
    save_file = os.path.join(save_path, "Tuning.txt")
    with open(save_file, 'a') as f:
        with redirect_stdout(f):
            print(params)
    print (params)
    model.fit(
        X_train, y_train,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        verbose=0,
        callbacks=callbacks,
    )
    
    # Evaluate model
    loss = model.evaluate(X_train, y_train, verbose=0)[0]
    if np.isfinite(loss):
        train_mse = mean_squared_error(y_scaler.inverse_transform(y_train),y_scaler.inverse_transform(model.predict(X_train)))
        val_mse = mean_squared_error(y_scaler.inverse_transform(y_val),y_scaler.inverse_transform(model.predict(X_val)))
    else:
        train_mse = np.inf
        val_mse = np.inf
    with open(save_file, 'a') as f:
        with redirect_stdout(f):
            print("Training MSE: {:.6f}, Validation MSE: {:.6f}\n".format(train_mse, val_mse))
    sys.stdout.flush() 
    
    return {'loss': val_mse, 'status': STATUS_OK, 'model': model}


start = time.time()
trials = Trials()
best_run = fmin(f_model, 
                space, 
                algo=tpe.rand.suggest, 
                max_evals=ntrials, 
                trials=trials, 
                return_argmin=False)

end = time.time()
best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
best_model.save_weights(os.path.join(save_path, 'model_opt'))

hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
with open(os.path.join(save_path, "Tuning.txt"), "a") as f:
    with redirect_stdout(f):
        print("Training completed in {:0>2}:{:0>2}:{:05.2f} for {} hyperparameter settings.\n".format(int(hours), int(minutes), seconds, ntrials))
print("Training completed in {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds, ntrials)) 


loss = best_model.evaluate(X_train, y_train, verbose=0)[0]
if np.isfinite(loss):
    train_mse, _ = utils.metrics(y_train, best_model.predict(X_train), y_preprocessor=y_scaler)
    val_mse, _ = utils.metrics(y_val, best_model.predict(X_val), y_preprocessor=y_scaler)
    test_mse, test_rmse = utils.metrics(y_test, best_model.predict(X_test), y_preprocessor=y_scaler)
else:
    train_mse = np.inf
    val_mse = np.inf
    test_mse = np.inf
    test_rmse = np.inf

df.loc['FC-NN','Training MSE'] = train_mse
df.loc['FC-NN','Validation MSE'] = val_mse
df.loc['FC-NN','Test MSE'] = test_mse
df.loc['FC-NN','Test RMSE'] = test_rmse
df.loc['FC-NN','Optimal Hyperparameters']=', '.join([f'{key}: {value}' for key, value in best_run.items()])
df.loc['FC-NN','Nonzeros'] = X.shape[1]
df.loc[['FC-NN'],:]        
with open(os.path.join(save_directory, 'results.csv'), 'a') as f:
    df.to_csv(f, header=True, sep='\t', encoding='utf-8', index=True)
