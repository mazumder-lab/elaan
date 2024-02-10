#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4-00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=shibal@mit.edu
#SBATCH --output=aoas_%A.out
#SBATCH --error=aoas_%A.err
#SBATCH --array=0-19

# Initialize the module command first
source /etc/profile

# Load modules
module load anaconda/2021a

# Call your script as you would from your command line
source activate aoas

export HDF5_USE_FILE_LOCKING=FALSE

# cd /home/gridsan/shibal/elaan/src/
# ELAAN
# /home/gridsan/shibal/.conda/envs/aoas/bin/python /home/gridsan/shibal/elaan/src/elaan/elaan_census.py --load_directory '/home/gridsan/shibal/elaan/Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1 --eval_criteria 'mse' --logging 

# ELAAN-I
# /home/gridsan/shibal/.conda/envs/aoas/bin/python /home/gridsan/shibal/elaan/src/elaani/elaani_census.py --load_directory '/home/gridsan/shibal/elaan/Census-Data' --seed $SLURM_ARRAY_TASK_ID --relative_penalty 1.0 --grid_search 'reduced' --run_first_round --version 1 --eval_criteria 'mse' --logging 

# ELAAN-H
# For ELAAN-H, we use gurobi to first solve convex relaxation. Make sure gurobipy is installed with a license
#module load gurobi/gurobi-903
# /home/gridsan/shibal/.conda/envs/elaan/bin/python /home/gridsan/shibal/elaan/src/elaanh/elaanh_census.py --load_directory '/home/gridsan/shibal/elaan/Census-Data' --seed $SLURM_ARRAY_TASK_ID --relative_penalty 1.0 --version 1 --eval_criteria 'mse' --logging