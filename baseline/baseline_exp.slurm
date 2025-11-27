#!/bin/bash
#SBATCH --job-name=baseline_exp
#SBATCH --partition=phil
#SBATCH --gres=gpu:1
#SBATCH --account=doriancl
#SBATCH --time=00:01:00

#SBATCH --mail-user=doriancl@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

# sacct -j <job_id> --format=JobID,JobName,State,Elapsed

/home/export/doriancl/code/Fall-2025-ML-proj/.venv/bin/python /home/export/doriancl/code/Fall-2025-ML-proj/baseline/baseline_server.py