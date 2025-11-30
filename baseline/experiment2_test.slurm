#!/bin/bash
#SBATCH --job-name=exp2test
#SBATCH --partition=phil
#SBATCH --cpus-per-task 32
#SBATCH --mem-per-cpu 4G
#SBATCH --gres=gpu:1
#SBATCH --account=doriancl
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/export/doriancl/code/Fall-2025-ML-proj/logs/%x-%A_%a.out

#SBATCH --mail-user=doriancl@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

task_id=${SLURM_ARRAY_TASK_ID}

# Define your sweep values here
PTYPES=("oneshot" "simple")
THINKS=(True False)

# Map 1D index → 2D indices (0-based)
i=$(( task_id / 2 ))   # which A index
j=$(( task_id % 2 ))   # which B index

PTYPE=${PTYPES[$i]}
THINK=${THINKS[$j]}

cd /home/export/doriancl/code/Fall-2025-ML-proj || exit 0
source .venv/bin/activate

echo "Task $SLURM_ARRAY_TASK_ID: prompt=$PTYPE"

srun python baseline/baseline_server.py \
  --language "ben" \
  --prompt-type "$PTYPE" \
  --model "Qwen/Qwen3-1.7B" \
  --max_concurrent "1024" \
  --thinking "$THINK"
