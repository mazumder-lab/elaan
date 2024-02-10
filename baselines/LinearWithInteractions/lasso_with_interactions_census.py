# Linear Models: Lasso with interactions baselines

# This notebook runs baseline linear models. Whenever we use more sophisticated learning models, we want to compare against a simpler baseline to make sure the computational effort is worth it.  

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

# Import Processed Data

parser = argparse.ArgumentParser(description='Lasso with Interactions Regression on Census data.')

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
save_directory = os.path.join(str(pathlib.Path(__file__).parent.absolute()).split('baselines')[0], "results", "LinearWithInteractions", "census", "v{}".format(args.version), "seed{}".format(args.seed), "lasso-with-interactions") 
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

_, d = X_train.shape
interaction_terms = []
for m in range(0,d):
    for n in range(0,d):
        if m!=n and m<n:
            interaction_terms.append((m,n))

X_train_interactions = np.zeros((X_train.shape[0], len(interaction_terms)))
X_val_interactions = np.zeros((X_val.shape[0], len(interaction_terms)))
X_test_interactions = np.zeros((X_val.shape[0], len(interaction_terms)))

for i, (m,n) in notebook.tqdm(enumerate(interaction_terms)):
    X_train_interactions[:, i] = X_train[:, m] * X_train[:, n]
    X_val_interactions[:, i] = X_val[:, m] * X_val[:, n]
    X_test_interactions[:, i] = X_test[:, m] * X_test[:, n]

X_train = np.hstack([X_train, X_train_interactions])
X_val = np.hstack([X_val, X_val_interactions])
X_test = np.hstack([X_test, X_test_interactions])
print(X_train.shape, X_val.shape, X_test.shape)

del X_train_interactions
del X_val_interactions
del X_test_interactions
import gc
gc.collect()

X = np.append(X_train, X_val, axis=0)
y = np.append(y_train, y_val, axis=0)
val_fold = np.append(-1*np.ones(y_train.shape), np.zeros(y_val.shape))
ps = PredefinedSplit(test_fold=val_fold)
neg_scorer = make_scorer(mean_squared_error, greater_is_better=False)

pen = np.array(['LassoWithInteractions'])
M = pen.shape[0]
df = pd.DataFrame(data={'': pen,
                      'Training MSE': np.zeros(M), 
                      'Validation MSE': np.zeros(M),
                      'Test MSE': np.zeros(M),
                      'Test RMSE': np.zeros(M),
                      'Nonzeros': np.zeros(M),
                      'MainEffects': np.zeros(M),
                      'InteractionEffects': np.zeros(M),
                       })
df = df.set_index('')

# Lasso Regression

model = Lasso()
alpha = np.logspace(start=-5, stop=1, num=100, base=10.0)
param_grid ={'alpha': alpha}
lasso_grid = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring=neg_scorer,
                          cv=ps,
                          n_jobs=1,
                          verbose=3)
lasso_grid.fit(X, np.ravel(y))

df.loc['LassoWithInteractions','Training MSE'], _ = utils.metrics(y_train, lasso_grid.best_estimator_.predict(X_train), y_preprocessor=y_scaler)
df.loc['LassoWithInteractions','Validation MSE'], _ = utils.metrics(y_val, lasso_grid.best_estimator_.predict(X_val), y_preprocessor=y_scaler)
optimal_solution = lasso_grid.best_estimator_.coef_
main_effects = optimal_solution[:d]
interaction_effects = optimal_solution[d:] 
df.loc['LassoWithInteractions','MainEffects'] = np.count_nonzero(main_effects)
df.loc['LassoWithInteractions','InteractionEffects'] = np.count_nonzero(interaction_effects)
df.loc['LassoWithInteractions','Nonzeros'] = np.union1d(np.nonzero(main_effects)[0], np.unique(np.array([interaction_terms[k] for k in np.nonzero(interaction_effects)[0]]))).shape[0]
df.loc['LassoWithInteractions','Test MSE'], df.loc['LassoWithInteractions','Test RMSE'] = utils.metrics(y_test, lasso_grid.best_estimator_.predict(X_test), y_preprocessor=y_scaler)
df.loc['LassoWithInteractions','Optimal Hyperparameters'] = ', '.join([f'{key}: {value}' for key, value in lasso_grid.best_params_.items()])  
# display(df.loc[['LassoWithInteractions'],:])
with open(os.path.join(save_directory, 'results.csv'), 'a') as f:
    df.loc[['LassoWithInteractions'],:].to_csv(f, header=True, sep='\t', encoding='utf-8', index=True)
with open(os.path.join(save_directory, 'lasso_with_interactions.pkl'), 'wb') as output:
    dill.dump(lasso_grid, output)

optimal_solution = lasso_grid.best_estimator_.coef_
main_effects = optimal_solution[:d]
interaction_effects = optimal_solution[d:] 
print("nnz(M):", np.count_nonzero(main_effects))
print("nnz(I):", np.count_nonzero(interaction_effects))
print("# covariates:", np.union1d(np.nonzero(main_effects)[0], np.unique(np.array([interaction_terms[k] for k in np.nonzero(interaction_effects)[0]]))).shape[0])

