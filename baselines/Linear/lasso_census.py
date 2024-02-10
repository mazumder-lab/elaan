import sys
import os
print(sys.version, sys.platform, sys.executable) #Displays what environment you are actually using.

import dill
import pandas as pd
import numpy as np
from tqdm import notebook
from copy import deepcopy

from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, PredefinedSplit

import data_utils
import utils

import pathlib
import argparse

parser = argparse.ArgumentParser(description='Lasso regression on Census data.')

# Data Arguments
parser.add_argument('--load_directory', dest='load_directory',  type=str, default='/home/shibal/Census-Data')
parser.add_argument('--seed', dest='seed',  type=int, default=8)

# Logging Arguments
parser.add_argument('--version', dest='version',  type=int, default=1)

args = parser.parse_args()

# # Import Processed Data

# Load directory needs to be updated with the path of the dropbox folder downloaded from the following link: 
# https://www.dropbox.com/sh/piwdz9sbmxjx03f/AACAKqjunrEhaRY9SORRE-Yba?dl=0

load_directory=args.load_directory
save_directory = os.path.join(str(pathlib.Path(__file__).parent.absolute()).split('baselines')[0], "results", "Linear", "v{}".format(args.version), "seed{}".format(args.seed), "lasso") 
os.makedirs(save_directory, exist_ok=True)

df_X, df_y, _ = data_utils.load_data(load_directory=load_directory,
                                  filename='pdb2019trv3_us.csv',
                                  remove_margin_of_error_variables=True)
seed = args.seed
np.random.seed(seed)
X_train, y_train, X_val, y_val, X_test, y_test, x_scaler, y_scaler = data_utils.process_data(
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

pen = np.array(['Lasso'])
M = pen.shape[0]
df = pd.DataFrame(data={'': pen,
                      'Training MSE': np.zeros(M), 
                      'Validation MSE': np.zeros(M),
                      'Test MSE': np.zeros(M),
                      'Test RMSE': np.zeros(M),
                      'Nonzeros': np.zeros(M)})
df = df.set_index('')

# Lasso Regression

model = Lasso()
alpha = np.logspace(start=-6, stop=1, num=500, base=10.0)
param_grid ={'alpha': alpha}
lasso_grid = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring=neg_scorer,
                          cv=ps,
                          n_jobs=2,
                          verbose=2)
lasso_grid.fit(X, np.ravel(y))
df.loc['Lasso','Training MSE'], _ = utils.metrics(y_train, lasso_grid.best_estimator_.predict(X_train), y_preprocessor=y_scaler)
df.loc['Lasso','Validation MSE'], _ = utils.metrics(y_val, lasso_grid.best_estimator_.predict(X_val), y_preprocessor=y_scaler)
df.loc['Lasso','Nonzeros'] = np.count_nonzero(lasso_grid.best_estimator_.coef_)
df.loc['Lasso','Test MSE'], df.loc['Lasso','Test RMSE'] = utils.metrics(y_test, lasso_grid.best_estimator_.predict(X_test), y_preprocessor=y_scaler)
df.loc['Lasso','Optimal Hyperparameters'] = ', '.join([f'{key}: {value}' for key, value in lasso_grid.best_params_.items()])  
# display(df.loc[['Lasso'],:])
with open(os.path.join(save_directory, 'results.csv'), 'a') as f:
    df.loc[['Lasso'],:].to_csv(f, header=True, sep='\t', encoding='utf-8', index=True)
with open(os.path.join(save_directory, 'lasso.pkl'), 'wb') as output:
    dill.dump(lasso_grid, output)
