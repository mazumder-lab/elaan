#!/bin/bash
#SBATCH -a 1-100
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00
#SBATCH --mem=32G
#SBATCH -o /home/gridsan/shibal/elaan/baselines/GamiNet/examples/logs/large-synthetic-correlated/normal/v1/seed%a_%j.out
#SBATCH -e /home/gridsan/shibal/elaan/baselines/GamiNet/examples/logs/large-synthetic-correlated/normal/v1/seed%a_%j.err

echo 'My SLURM_ARRAY_TASK_ID: ' $SLURM_ARRAY_TASK_ID
echo 'Number of Tasks: ' $SLURM_ARRAY_TASK_COUNT

/home/gridsan/shibal/.conda/envs/baselines/bin/python -u ./gaminet_synthetic.py  --dist 'normal' --seed $SLURM_ARRAY_TASK_ID --dataset 'large-synthetic-correlated' --correlation 0.7 --train_size 10000 --version 1