#!/bin/bash
#SBATCH --job-name=E2-Q3-14-Think
#SBATCH --partition=phil
#SBATCH --cpus-per-task 32
#SBATCH --mem-per-cpu 4G
#SBATCH --gres=gpu:1
#SBATCH --account=doriancl
#SBATCH --time=28:00:00
#SBATCH --output=/home/export/doriancl/code/Fall-2025-ML-proj/logs/%x-%A_%a.out

#SBATCH --mail-user=doriancl@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

task_id=8

# Define your sweep values here
LANGS=("ber" "cmn" "deu" "ita" "jpn" "tur" "fra")
PTYPES=("iterative" "monolithic")

N_LANGS=${#LANGS[@]}
N_PTYPES=${#PTYPES[@]}

i=$(( task_id / N_PTYPES ))
j=$(( task_id % N_PTYPES ))

LANG=${LANGS[$i]}
PTYPE=${PTYPES[$j]}

cd /home/export/doriancl/code/Fall-2025-ML-proj || exit 0
source .venv/bin/activate

echo "Rerunning task $SLURM_ARRAY_TASK_ID: lang=$LANG"

srun python baseline/baseline_server.py \
  --language "$LANG" \
  --prompt-type "$PTYPE" \
  --model "Qwen/Qwen3-14B" \
  --max_concurrent "768" \
  --thinking true \
  --n_samples 45000
