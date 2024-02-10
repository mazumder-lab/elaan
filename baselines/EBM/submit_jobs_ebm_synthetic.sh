#!/bin/bash
#SBATCH -a 1-25
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00
#SBATCH --mem=32G
#SBATCH -o /home/gridsan/shibal/elaan/baselines/EBM/logs/synthetic/normal/v1/seed%a_%j.out
#SBATCH -e /home/gridsan/shibal/elaan/baselines/EBM/logs/synthetic/normal/v1/seed%a_%j.err

echo 'My SLURM_ARRAY_TASK_ID: ' $SLURM_ARRAY_TASK_ID
echo 'Number of Tasks: ' $SLURM_ARRAY_TASK_COUNT

/home/gridsan/shibal/.conda/envs/EBM/bin/python -u ./ebm_synthetic.py  --dist 'normal' --seed $SLURM_ARRAY_TASK_ID --dataset 'synthetic' --correlation 0.5 --train_size 100 --version 1 