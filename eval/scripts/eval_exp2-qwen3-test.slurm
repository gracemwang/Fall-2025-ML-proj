#!/bin/bash
#SBATCH --job-name=eval_test
#SBATCH --partition=statds_condo
#SBATCH --cpus-per-task 30
#SBATCH --mem-per-cpu 4G
#SBATCH --gres=gpu:1
#SBATCH --account=doriancl
#SBATCH --time=1:00:00
#SBATCH --array=0 # 0-13
#SBATCH --output=/home/export/doriancl/code/Fall-2025-ML-proj/logs/%x-%A_%a.out

#SBATCH --mail-user=doriancl@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

task_id=${SLURM_ARRAY_TASK_ID}

# Define your sweep values here
LANGS=("ber" "cmn" "deu" "ita" "jpn" "tur" "fra")
PTYPES=("iterative" "monolithic")
TTYPES=("true" "false")

N_LANGS=${#LANGS[@]}
N_PTYPES=${#PTYPES[@]}
N_TTYPES=${#TTYPES[@]}

i=$(( task_id / N_PTYPES ))
j=$(( task_id % N_PTYPES ))

LANG=${LANGS[$i]}
PTYPE=${PTYPES[$j]}

cd /home/export/doriancl/code/Fall-2025-ML-proj || exit 0
source .venv/bin/activate

echo "Task $SLURM_ARRAY_TASK_ID: lang=$LANG"

srun python eval/score_translations.py \
  --language "$LANG" \
  --prompt-type "$PTYPE" \
  --model "Qwen/Qwen3-1.7B" \
  --thinking false \
  --n_samples 500