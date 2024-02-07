# AMs with interactions under L0

# $$
# \begin{align}
#     \min_{\beta_i, \theta_{ij}} & \frac{1}{2N}\left\lVert y - \left[\sum_{i}^{p} B_i \beta_i + \sum_{ij}^{p \choose 2} B_{ij} \theta_{ij} \right]\right\rVert_2^2
#     + \lambda_1 \left[\sum_{i}^{p} \beta_i^T S_i \beta_i + \sum_{ij}^{p \choose 2} \theta_{ij}^T S_{ij} \theta_{ij} \right] +  \lambda_2 \left[\sum_{i}^{p} \mathbf{1}[\beta_i \neq 0] + \alpha \sum_{ij}^{p \choose 2} \mathbf{1}[\theta_{ij} \neq 0] \right] \\
# \end{align}
# $$

from __future__ import division
import sys
print(sys.version, sys.platform, sys.executable) #Displays what environment you are actually using.

from copy import deepcopy

import dill
import gc
from ipywidgets import *
import numpy as np
import pandas as pd
from scipy.special import comb

import pathlib
import argparse

sys.path.insert(0, os.path.abspath(str(pathlib.Path(__file__).absolute())).split('src')[0])
from src import data_utils
from src import utils
from src.elaani import models


parser = argparse.ArgumentParser(description='ELAAN-I: Additive Models with Interactions under L0 on Census data.')

# Data Arguments
parser.add_argument('--load_directory', dest='load_directory',  type=str, default='/home/shibal/Census-Data')
parser.add_argument('--seed', dest='seed',  type=int, default=8)

# Logging Arguments
parser.add_argument('--version', dest='version',  type=int, default=1)

# Model Arguments
parser.add_argument('--Ki', dest='Ki',  type=int, default=10)
parser.add_argument('--Kij', dest='Kij',  type=int, default=5)
parser.add_argument('--relative_penalty', dest='r',  type=float, default=1.0)

# Algorithm Arguments
parser.add_argument('--max_interaction_support', dest='max_interaction_support',  type=int, default=400) # cuts off the L0 regularization path when the number of interactions reach 400.
parser.add_argument('--convergence_tolerance', dest='convergence_tolerance',  type=float, default=1e-4)
parser.add_argument('--grid_search', dest='grid_search',  type=str, default='full') # 'full', 'reduced'
# parser.add_argument('--load_model', dest='load_model', action='store_true')
# parser.add_argument('--no_load_model', dest='load_model', action='store_false') # only used when grid_search is 'reduced'
# parser.set_defaults(load_model=False)
parser.add_argument('--run_first_round', dest='run_first_round', action='store_true')
parser.add_argument('--no_run_first_round', dest='run_first_round', action='store_false')
parser.set_defaults(run_first_round=False)
parser.add_argument('--run_second_round', dest='run_second_round', action='store_true')
parser.add_argument('--no_run_second_round', dest='run_second_round', action='store_false')
parser.set_defaults(run_second_round=False)


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

### Initialize parameters

convergence_tolerance = args.convergence_tolerance
column_names = df_X.columns
r = args.r
logging = args.logging
max_interaction_support=args.max_interaction_support
version = args.version

# for c in column_names:
#     print(c)

### How to run the model
# Running the model over a 2 dimensional grid of hyperparameters (smoothing: $\lambda_1$, L0: $\lambda_2$) on ~300 covariates and ~45000 interaction effects requires large memory/time resources. 

# We have 3 cases:
# 1) Run model on all interaction effects

# 2) Run model using Top-I interaction effects

# 3) Run model using union of interaction effects recovered from (1) over the hyperparameter grid search. This either requires a saved model of (1) or a defined 2D array of interaction effects.

Ki = args.Ki
Kij = args.Kij
eval_criteria = args.eval_criteria
p = X.shape[1]
X = X[:, :p]
Xval = Xval[:, :p]
Xtest = Xtest[:, :p]
N, _ = X.shape
Xmin = np.min(np.vstack([X, Xval, Xtest]), axis=0)
Xmax = np.max(np.vstack([X, Xval, Xtest]), axis=0)
lams_sm_start = -3
lams_sm_stop = -6
lams_L0_start = 1
lams_L0_stop = -4
lams_sm=np.logspace(start=lams_sm_start, stop=lams_sm_stop, num=20, base=10.0)
lams_L0=np.logspace(start=lams_L0_start, stop=lams_L0_stop, num=100, base=10.0)

if args.run_first_round:
    path = os.path.join(save_directory, 'ELAANI', 'v{}'.format(version), 'r{}'.format(r), 'seed{}'.format(seed), 'firstround')
    os.makedirs(path, exist_ok=True)

    with open(path+'/Parameters.txt', "w") as f:
        [f.write('{}: {}\n'.format(k,v)) for k,v in vars(args).items()]
        f.write('Train: {}, Validation: {}, Test: {}\n'.format(X.shape[0], Xval.shape[0], Xtest.shape[0])) 
    
    am = models.ELAANI(
        lams_sm=lams_sm,
        lams_L0=lams_L0,
        alpha=r,
        convergence_tolerance=convergence_tolerance,
        eval_criteria=eval_criteria,
        path=path,
        max_interaction_support=max_interaction_support,
        terminate_val_L0path=False
    )
    am.load_data(X, Y, y_scaler, column_names, Xmin, Xmax)

    grid_search = args.grid_search
#     load_model = args.load_model
    if grid_search=='full':
        am.generate_interaction_terms(generate_all_pairs=True)
    elif grid_search=='reduced':
        generate_all_pairs = False
#         if load_model:
#             folder = 'ELAANI/v1.0/r1.0'
#             with open(os.path.join(save_directory, folder, 'model.pkl'), 'rb') as input:
#                 ami_loaded = dill.load(input)
#             # active_set = ami_loaded.active_set_union
#             interaction_terms = ami_loaded.interaction_terms_union
#             # active_set = np.sort(np.union1d(active_set, np.unique(interaction_terms)))
#             # print("Number of main effects to consider:", len(active_set)) 
#             am.interaction_terms = interaction_terms
#             am.generate_all_pairs = False
#             am.I = (int)(comb(am.p, 2, exact=False))
#             am.Imax = len(interaction_terms)
#         else:
        # consider Top-I interaction terms to consider [based on marginal fits]
        am.generate_interaction_terms(generate_all_pairs=False, Imax=10000)
    print("Number of interaction effects to consider:", len(am.interaction_terms))

    am.generate_splines_and_quadratic_penalties(Ki=Ki, Kij=Kij)
    am.fit_validation_path(Xval, Yval)
    am.evaluate_and_save(Xtest, Ytest)
    am.Btrain = None
    am.BtrainT_Btrain = None
    am.Btrain_interaction = None
    am.Btrain_interactionT_Btrain_interaction = None
    with open(os.path.join(path, 'model.pkl'), 'wb') as output:
        dill.dump(am, output)

# Second pass over the hyperparameter grid is much faster as it considers only the reduced set of promising interaction effects that appeared in the solutions over the grid. Due to nonconvexity of $\ell_0$-norm, the solution quality may improve with this second run! we did not consider this in the paper.

if args.run_second_round:
    path = os.path.join(save_directory, 'ELAANI', 'v{}'.format(version), 'r{}'.format(r), 'seed{}'.format(seed), 'secondround')
    os.makedirs(path, exist_ok=True)

    with open(path+'/Parameters.txt', "w") as f:
        [f.write('{}: {}\n'.format(k,v)) for k,v in vars(args).items()]
        f.write('Train: {}, Validation: {}, Test: {}\n'.format(X.shape[0], Xval.shape[0], Xtest.shape[0])) 
    
    am_new = models.ELAANI(
        lams_sm=lams_sm,
        lams_L0=lams_L0,
        alpha=r,
        convergence_tolerance=convergence_tolerance,
        eval_criteria=eval_criteria,
        path=path,
        max_interaction_support=max_interaction_support,
        terminate_val_L0path=False
    )
    am_new.load_data(X, Y, y_scaler, column_names, Xmin, Xmax)

    load_path = os.path.join(save_directory, 'ELAANI', 'v{}'.format(version), 'r{}'.format(r), 'seed{}'.format(seed), 'firstround')
    with open(os.path.join(load_path, 'model.pkl'), 'rb') as input:
        ami_loaded = dill.load(input)
    # active_set = ami_loaded.active_set_union
    interaction_terms = ami_loaded.interaction_terms_union
    # active_set = np.sort(np.union1d(active_set, np.unique(interaction_terms)))
    # print("Number of main effects to consider:", len(active_set)) 
    print("Number of interaction effects to consider:", len(interaction_terms)) 
    
    am_new.interaction_terms = interaction_terms
    am_new.generate_all_pairs = False
    am_new.I = (int)(comb(p, 2, exact=False))
    am_new.Imax = len(interaction_terms)

    am_new.generate_splines_and_quadratic_penalties(Ki=Ki, Kij=Kij)
    am_new.fit_validation_path(Xval, Yval)
    am_new.evaluate_and_save(Xtest, Ytest)
    am_new.Btrain = None
    am_new.BtrainT_Btrain = None
    am_new.Btrain_interaction = None
    am_new.Btrain_interactionT_Btrain_interaction = None
    with open(os.path.join(path, 'model_final.pkl'), 'wb') as output:
        dill.dump(am_new, output)
