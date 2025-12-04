#!/bin/bash
#SBATCH --job-name=eval_test
#SBATCH --partition=phil
#SBATCH --cpus-per-task 30
#SBATCH --mem-per-cpu 4G
#SBATCH --gres=gpu:1
#SBATCH --account=doriancl
#SBATCH --time=0:05:00
#SBATCH --array=0-55
#SBATCH --output=/home/export/doriancl/code/Fall-2025-ML-proj/logs/%x-%A_%a.out
#SBATCH --error=/home/export/doriancl/code/Fall-2025-ML-proj/logs/%x-%A_%a.err


#SBATCH --mail-user=doriancl@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

task_id=${SLURM_ARRAY_TASK_ID}

# Define your sweep values here
LANGS=("ber" "cmn" "deu" "ita" "jpn" "tur" "fra") # length m
PTYPES=("iterative" "monolithic")                 # length n
TTYPES=("true" "false")                           # length o
MTYPES=("Qwen/Qwen3-14B" "Qwen/Qwen3-1.7B")       # length p

M=${#LANGS[@]}
N=${#PTYPES[@]}
O=${#TTYPES[@]}
P=${#MTYPES[@]}

TOTAL=$((M * N * O * P))

if (( task_id >= TOTAL )); then
    echo "Task ID $task_id >= total combos $TOTAL — exiting."
    exit 0
fi

# ---- MAP task_id → (i, j, k) ----
# i ∈ [0, M)
# j ∈ [0, N)
# k ∈ [0, O)
# l ∈ [0, P)

block_BC=$((N * O * P))
i=$(( task_id / block_BC ))
rem=$(( task_id % block_BC ))

block_C=$((O * P))
j=$(( rem / block_C ))
rem=$(( rem % block_C ))

k=$(( rem / P ))
l=$(( rem % P ))

LANG=${LANGS[$i]}
PROMPT=${PTYPES[$j]}
THINK=${TTYPES[$k]}
MODEL=${MTYPES[$l]}


cd /home/export/doriancl/code/Fall-2025-ML-proj || exit 0
source .venv/bin/activate

echo "Task $SLURM_ARRAY_TASK_ID: lang=$LANG, prompt=$PROMPT, thinking=$THINK, model=$MODEL"

srun python eval/score_translations.py \
  --language "$LANG" \
  --prompt-type "$PROMPT" \
  --model "$MODEL" \
  --thinking $THINK \
  --n_samples 100