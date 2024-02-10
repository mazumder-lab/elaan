# AMs L0

from __future__ import division

import sys
print(sys.version, sys.platform, sys.executable) #Displays what environment you are actually using.

import dill
import gc

from copy import deepcopy
from IPython.display import Math
from ipywidgets import *
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
from tqdm import notebook

import pathlib
import argparse

sys.path.insert(0, os.path.abspath(str(pathlib.Path(__file__).absolute())).split('src')[0])
from src import data_utils
from src import utils
from src.elaan import models


parser = argparse.ArgumentParser(description='ELAAN: Additive Models with only main effects under L0 on Census data.')

# Data Arguments
parser.add_argument('--load_directory', dest='load_directory',  type=str, default='/home/shibal/Census-Data')
parser.add_argument('--seed', dest='seed',  type=int, default=8)

# Logging Arguments
parser.add_argument('--version', dest='version',  type=int, default=1)

# Model Arguments
parser.add_argument('--Ki', dest='Ki',  type=int, default=10)

# Algorithm Arguments
parser.add_argument('--convergence_tolerance', dest='convergence_tolerance',  type=float, default=1e-4)


# Tuning Arguments
parser.add_argument('--eval_criteria', dest='eval_criteria',  type=str, default='mse')

# Logging Arguments
parser.add_argument('--logging', dest='logging', action='store_true')
parser.add_argument('--no-logging', dest='logging', action='store_false')
parser.set_defaults(logging=True)

args = parser.parse_args()

# Import Processed Data

load_directory=args.load_directory
save_directory = os.path.join(os.path.abspath(str(pathlib.Path(__file__).absolute())).split('src')[0], "results") 

df_X, df_y, _ = data_utils.load_data(load_directory=load_directory,
                                  filename='pdb2019trv3_us.csv',
                                  remove_margin_of_error_variables=True)
seed = args.seed
np.random.seed(seed)
X, Y, Xval, Yval, Xtest, Ytest, _, y_scaler = data_utils.process_data(
    df_X,
    df_y,
    val_ratio=0.1, 
    test_ratio=0.1,
    seed=seed,
    standardize_response=False)

### Initialize Parameters

convergence_tolerance = args.convergence_tolerance
column_names = df_X.columns
max_support=X.shape[1]
logging = args.logging
version = args.version

path = os.path.join(save_directory, 'ELAAN', 'v{}'.format(version), 'seed{}'.format(seed))
os.makedirs(path, exist_ok=True)

with open(path+'/Parameters.txt', "w") as f:
    [f.write('{}: {}\n'.format(k,v)) for k,v in vars(args).items()]
    f.write('Train: {}, Validation: {}, Test: {}\n'.format(X.shape[0], Xval.shape[0], Xtest.shape[0])) 

# for c in column_names:
#     print(c)

Ki = 10
p = X.shape[1]
N, _ = X.shape
Xmin = np.min(np.vstack([X, Xval, Xtest]), axis=0)
Xmax = np.max(np.vstack([X, Xval, Xtest]), axis=0)
lams_sm_start = -2
lams_sm_stop = -6
lams_L0_start = 1
lams_L0_stop = -5
lams_sm=np.logspace(start=lams_sm_start, stop=lams_sm_stop, num=20, base=10.0)
lams_L0=np.logspace(start=lams_L0_start, stop=lams_L0_stop, num=100, base=10.0)
am = models.ELAAN(
    lams_sm=lams_sm,
    lams_L0=lams_L0,
    convergence_tolerance=convergence_tolerance,
    path=path,
    max_support=max_support,
    terminate_val_L0path=False
)
am.load_data(X, Y, y_scaler, column_names, Xmin, Xmax)
am.generate_main_terms()
am.generate_splines_and_quadratic_penalties(Ki=Ki)
am.fit_validation_path(Xval, Yval)
am.evaluate_and_save(Xtest, Ytest)
am.Btrain = None
am.BtrainT_Btrain = None
with open(os.path.join(path, 'model.pkl'), 'wb') as output:
    dill.dump(am, output)

