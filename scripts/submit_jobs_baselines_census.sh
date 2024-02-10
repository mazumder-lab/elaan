#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4-00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=shibal@mit.edu
#SBATCH --output=job_%A_%a.out
#SBATCH --error=job_%A_%a.err
#SBATCH --array=0-19

# Initialize the module command first
source /etc/profile

# Load modules
module load anaconda/2021a
#module load gurobi/gurobi-903

# Call your script as you would from your command line
source activate aoas

export HDF5_USE_FILE_LOCKING=FALSE

########## Additive Models with Interactions e.g., EBM, GamiNet ################################
# GamiNet
# cd /home/gridsan/shibal/elaan/baselines/GamiNet/examples/
# /home/gridsan/shibal/.conda/envs/additive2/bin/python /home/gridsan/shibal/elaan/baselines/GamiNet/examples/gaminet_census.py --load_directory '/home/gridsan/shibal/elaan/Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1

# EBM
# cd /home/gridsan/shibal/elaan/baselines/EBM/
# /home/gridsan/shibal/.conda/envs/EBM/bin/python /home/gridsan/shibal/elaan/baselines/EBM/ebm_census.py --load_directory '/home/gridsan/shibal/elaan/Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1 --ntrials 500

################################ Some other baselines ##########################################
########### Linear Models e.g., Ridge, Lasso
# /home/gridsan/shibal/.conda/envs/aoas/bin/python /home/gridsan/shibal/elaan/baselines/Linear/ridge_census.py --load_directory '/home/gridsan/shibal/Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1

# Linear Models e.g., Ridge, Lasso
# /home/gridsan/shibal/.conda/envs/aoas/bin/python /home/gridsan/shibal/elaan/baselines/Linear/lasso_census.py --load_directory '/home/gridsan/shibal/Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1

############ Linear models with interactions with Lasso
# /home/gridsan/shibal/.conda/envs/aoas/bin/python /home/gridsan/shibal/elaan/baselines/Linear/lasso_with_interactions_census.py --load_directory '/home/gridsan/shibal/Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1

############ BlackBox Methods
# XGBoost
# /home/gridsan/shibal/.conda/envs/aoas/bin/python -u /home/gridsan/shibal/elaan/baselines/BlackBox/xgboost_census.py --load_directory '/home/gridsan/shibal/Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1

# Feedforward Neural Networks
# /home/gridsan/shibal/.conda/envs/additive2/bin/python -u /home/gridsan/shibal/elaan/baselines/BlackBox/fcnn_census.py --load_directory '/home/gridsan/shibal/Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1
