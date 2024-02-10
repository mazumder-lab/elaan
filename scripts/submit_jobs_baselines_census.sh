#!/bin/bash

# SBATCH --cpus-per-task=8
# SBATCH --mem=128G
# SBATCH --time=4-00:00
# SBATCH --mail-type=FAIL
# SBATCH --mail-user=shibal@mit.edu
# SBATCH --output=aoas_%A.out
# SBATCH --error=aoas_%A.err
# SBATCH --array=0-0

# Initialize the module command first
source /etc/profile

# Load modules
module load anaconda/2021a
#module load gurobi/gurobi-903

# Call your script as you would from your command line
source activate aoas

export HDF5_USE_FILE_LOCKING=FALSE

# GAMINet
# cd /home/gridsan/shibal/elaan/baselines/GamiNet/examples/
# /home/gridsan/shibal/.conda/envs/additive2/bin/python /home/gridsan/shibal/elaan/baselines/GamiNet/examples/gaminet_census.py --load_directory '/home/gridsan/shibal/elaan/Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1

# EBM
# cd /home/gridsan/shibal/elaan/baselines/EBM/
# /home/gridsan/shibal/.conda/envs/EBM/bin/python /home/gridsan/shibal/elaan/baselines/EBM/ebm_census.py --load_directory '/home/gridsan/shibal/elaan/Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1 --ntrials 500


# cd /home/gridsan/shibal/Interpretable-Nonparametric-Additive-Models-with-Structured-Interactions/SparseGAMsWithInteractions/src/
# Ridge
# /home/gridsan/shibal/.conda/envs/aoas/bin/python /home/gridsan/shibal/Interpretable-Nonparametric-Additive-Models-with-Structured-Interactions/SparseGAMsWithInteractions/src/linear-ridge.py --load_directory '/home/gridsan/shibal/elaan/Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1 

# Lasso
# /home/gridsan/shibal/.conda/envs/aoas/bin/python /home/gridsan/shibal/Interpretable-Nonparametric-Additive-Models-with-Structured-Interactions/SparseGAMsWithInteractions/src/linear-lasso.py --load_directory '/home/gridsan/shibal/elaan/Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1 

# L0Learn
# /home/gridsan/shibal/.conda/envs/aoas/bin/python /home/shibal/Interpretable-Nonparametric-Additive-Models-with-Structured-Interactions/SparseGAMsWithInteractions/src/linear-l0l2.py --load_directory '/home/gridsan/shibal/elaan/Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1 

# Lasso With Interactions
# /home/gridsan/shibal/.conda/envs/aoas/bin/python /home/gridsan/shibal/Interpretable-Nonparametric-Additive-Models-with-Structured-Interactions/SparseGAMsWithInteractions/src/linear-lasso-with-interactions.py --load_directory '/home/gridsan/shibal/elaan/Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1 

# Lasso with Splines
# /home/gridsan/shibal/.conda/envs/aoas/bin/python /home/gridsan/shibal/Interpretable-Nonparametric-Additive-Models-with-Structured-Interactions/SparseGAMsWithInteractions/src/additive-models-lasso.py --load_directory '/home/gridsan/shibal/elaan/Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1

# XGBoost
# /home/gridsan/shibal/.conda/envs/aoas/bin/python /home/gridsan/shibal/Interpretable-Nonparametric-Additive-Models-with-Structured-Interactions/SparseGAMsWithInteractions/src/xgboost_tuning.py --load_directory '/home/gridsan/shibal/elaan/Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1 


# FCNN
# /home/gridsan/shibal/.conda/envs/aoas/bin/python /home/gridsan/shibal/Interpretable-Nonparametric-Additive-Models-with-Structured-Interactions/SparseGAMsWithInteractions/src/fcnn_tuning.py --load_directory '/home/gridsan/shibal/elaan/Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1
