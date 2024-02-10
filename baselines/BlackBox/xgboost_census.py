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

from sklearn.metrics import mean_absolute_error, mean_squared_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, SparkTrials
import tensorflow.keras.backend as K

import time
from copy import deepcopy
from tqdm import notebook
from ipywidgets import *

import xgboost as xgb

import data_utils
import utils
import pathlib
import argparse

parser = argparse.ArgumentParser(description='XGBoost on Census data.')

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
save_directory = os.path.join(str(pathlib.Path(__file__).parent.absolute()).split('baselines')[0], "results", "BlackBox", "v{}".format(version), "seed{}".format(seed), "xgboost") 

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
pen = np.array(['XGBoost'])
M = pen.shape[0]
df = pd.DataFrame(data={'': pen,
                      'Training MSE': np.zeros(M), 
                      'Validation MSE': np.zeros(M),
                      'Test MSE': np.zeros(M),
                      'Test RMSE': np.zeros(M),
                      'Nonzeros': np.zeros(M)})
df = df.set_index('')



# Uses hyperopt library to tune XGBoost

model = xgb.XGBRegressor(objective='reg:squarederror',
                         booster='gbtree',
#                          colsample_bytree=1.0,
                         eval_metric='rmse')
param_grid = {
              'n_estimators': np.arange(10, 200),
              'learning_rate': np.logspace(start=-4, stop=0, num=20, base=10),
              'max_depth': np.arange(1, 10),
             }

xgb_grid = RandomizedSearchCV(estimator=model,
                              param_distributions=param_grid,
                              scoring=neg_scorer,
                              n_iter=ntrials,
                              cv=PredefinedSplit(test_fold=val_fold),
                              n_jobs=2,
                              verbose=3)

xgb_grid.fit(X, np.ravel(y))
# xgb_grid.best_estimator_.save_model(os.path.join(save_directory, 'xgb_opt.model'))
df.loc['XGBoost','Training MSE'], _ = utils.metrics(y_train, xgb_grid.best_estimator_.predict(X_train), y_preprocessor=y_scaler)
df.loc['XGBoost','Validation MSE'], _ = utils.metrics(y_val, xgb_grid.best_estimator_.predict(X_val), y_preprocessor=y_scaler)
df.loc['XGBoost','Test MSE'], df.loc['XGBoost','Test RMSE'] = utils.metrics(y_test, xgb_grid.best_estimator_.predict(X_test), y_preprocessor=y_scaler)
df.loc['XGBoost','Optimal Hyperparameters']=', '.join([f'{key}: {value}' for key, value in xgb_grid.best_params_.items()])
df.loc['XGBoost','Nonzeros'] = np.sum(xgb_grid.best_estimator_.feature_importances_>0)
df.loc[['XGBoost'], :]

with open(os.path.join(save_directory, 'results.csv'), 'a') as f:
    df.to_csv(f, header=True, sep='\t', encoding='utf-8', index=True)
