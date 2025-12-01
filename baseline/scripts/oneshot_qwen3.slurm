#!/bin/bash
#SBATCH --job-name=1shot3
#SBATCH --partition=phil
#SBATCH --cpus-per-task 32
#SBATCH --mem-per-cpu 4G
#SBATCH --gres=gpu:1
#SBATCH --account=doriancl
#SBATCH --time=06:00:00
#SBATCH --array=0-6
#SBATCH --output=/home/export/doriancl/code/Fall-2025-ML-proj/logs/%x-%A_%a.out

#SBATCH --mail-user=doriancl@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

# Define your sweep values here
LANGS=("ber" "cmn" "deu" "ita" "jpn" "tur" "fra")

LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}

cd /home/export/doriancl/code/Fall-2025-ML-proj || exit 0
source .venv/bin/activate

echo "Task $SLURM_ARRAY_TASK_ID: lang=$LANG"

srun python baseline/baseline_server.py \
  --language "$LANG" \
  --prompt-type "oneshot" \
  --model "Qwen/Qwen3-14B"
