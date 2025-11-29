#!/bin/bash
#SBATCH --job-name=baseline_exp
#SBATCH --partition=phil
#SBATCH --cpus-per-task 32
#SBATCH --mem-per-cpu 4G
#SBATCH --gres=gpu:1
#SBATCH --account=doriancl
#SBATCH --time=05:00:00
#SBATCH --array=0-5           # 6 jobs: IDs 0,1,2,3,4,5
#SBATCH --output=/home/export/doriancl/code/Fall-2025-ML-proj/logs/%x-%A_%a.out

#SBATCH --mail-user=doriancl@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

# Define your sweep values here
LANGS=("ber" "cmn" "deu" "ita" "jpn" "tur")

LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}

cd /home/export/doriancl/code/Fall-2025-ML-proj || exit 0
source .venv/bin/activate

echo "Task $SLURM_ARRAY_TASK_ID: lang=$LANG"

srun python baseline/baseline_server.py \
  --language "$LANG" \
  --model "Qwen/Qwen3-14B"
