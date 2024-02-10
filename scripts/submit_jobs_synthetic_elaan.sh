#!/bin/bash
# SBATCH -a 1-100
# SBATCH --cpus-per-task=8
# SBATCH --time=4-00:00
# SBATCH --mem=32G
# SBATCH -o /home/gridsan/shibal/logs/seed%a_%j.out
# SBATCH -e /home/gridsan/shibal/logs/seed%a_%j.err

echo 'My SLURM_ARRAY_TASK_ID: ' $SLURM_ARRAY_TASK_ID
echo 'Number of Tasks: ' $SLURM_ARRAY_TASK_COUNT

module load anaconda/2021a
source activate aoas

# ELAAN-I
/home/gridsan/shibal/.conda/envs/aoas/bin/python /home/gridsan/shibal/elaan/src/elaani/elaani_synthetic.py  --dataset 'synthetic' --dist 'normal' --correlation 0.5 --seed $SLURM_ARRAY_TASK_ID --train_size 100 --version 1 --r 1.0 --Ki 10 --Kij 5

# ELAAN-H
# For ELAAN-H, we use gurobi to solve convex relaxation. Make sure gurobipy is installed with a license
#module load gurobi/gurobi-903
/home/gridsan/shibal/.conda/envs/aoas/bin/python /home/gridsan/shibal/elaan/src/elaanh/elaanh_synthetic.py  --dataset 'synthetic' --dist 'normal' --correlation 0.5 --seed $SLURM_ARRAY_TASK_ID --train_size 100 --version 1 --r 1.0 --Ki 10 --Kij 5
