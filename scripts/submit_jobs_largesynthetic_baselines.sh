#!/bin/bash
# SBATCH -a 1-100
# SBATCH --cpus-per-task=8
# SBATCH --time=4-00:00
# SBATCH --mem=32G
# SBATCH -o /home/gridsan/shibal/logs/seed%a_%j.out
# SBATCH -e /home/gridsan/shibal/logs/seed%a_%j.err

echo 'My SLURM_ARRAY_TASK_ID: ' $SLURM_ARRAY_TASK_ID
echo 'Number of Tasks: ' $SLURM_ARRAY_TASK_COUNT

# EBM
/home/gridsan/shibal/.conda/envs/EBM/bin/python -u /home/gridsan/shibal/elaan/baselines/EBM/ebm_synthetic.py  --dist 'normal' --seed $SLURM_ARRAY_TASK_ID --dataset 'large-synthetic-correlated' --correlation 0.7 --train_size 10000 --version 1 

# GamiNet
# /home/gridsan/shibal/.conda/envs/additive2/bin/python -u /home/gridsan/shibal/elaan/baselines/GamiNet/examples/gaminet_synthetic.py  --dist 'normal' --seed $SLURM_ARRAY_TASK_ID --dataset 'large-synthetic-correlated' --correlation 0.7 --train_size 10000 --version 1
