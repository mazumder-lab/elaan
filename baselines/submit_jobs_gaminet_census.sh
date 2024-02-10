#!/bin/bash
#SBATCH -a 1-20
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00
#SBATCH --mem=64G
#SBATCH -o /home/gridsan/shibal/elaan/baselines/GamiNet/examples/logs/census/v1/seed%a_%j.out
#SBATCH -e /home/gridsan/shibal/elaan/baselines/GamiNet/examples/logs/census/v1/seed%a_%j.err

echo 'My SLURM_ARRAY_TASK_ID: ' $SLURM_ARRAY_TASK_ID
echo 'Number of Tasks: ' $SLURM_ARRAY_TASK_COUNT

/home/gridsan/shibal/.conda/envs/baselines/bin/python ./gaminet_census.py --load_directory '../../Census-Data' --seed $SLURM_ARRAY_TASK_ID --version 1