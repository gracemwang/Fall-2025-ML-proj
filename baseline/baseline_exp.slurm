#!/bin/bash
#SBATCH --job-name=baseline_exp
#SBATCH --partition=cmist_condo
#SBATCH --cpus-per-task 32
#SBATCH --mem-per-cpu 4G
#SBATCH --gres=gpu:1
#SBATCH --account=doriancl
#SBATCH --time=05:00:00

##SBATCH --mail-user=doriancl@andrew.cmu.edu
##SBATCH --mail-type=END,FAIL

# sacct -j <job_id> --format=JobID,JobName,State,Elapsed

cd /home/export/doriancl/code/Fall-2025-ML-proj
source .venv/bin/activate

srun python baseline/baseline_server.py
