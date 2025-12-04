#!/bin/bash
#SBATCH --job-name=evalQ3
#SBATCH --partition=phil
#SBATCH --cpus-per-task 32
#SBATCH --mem-per-cpu 4G
#SBATCH --gres=gpu:1
#SBATCH --account=doriancl
#SBATCH --time=04:00:00
#SBATCH --array=0-6
#SBATCH --output=/home/export/doriancl/code/Fall-2025-ML-proj/logs/%x-%A_%a.out

# Define your sweep values here
LANGS=("ber" "cmn" "deu" "ita" "jpn" "tur" "fra")

LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}

cd /home/export/doriancl/code/Fall-2025-ML-proj || exit 0
source .venv/bin/activate

echo "Task $SLURM_ARRAY_TASK_ID: lang=$LANG"

srun python eval/score_translations.py \
  --language "$LANG" \
  --prompt-type "oneshot" \
  --model "Qwen/Qwen3-14B"
