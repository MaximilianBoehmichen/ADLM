#!/bin/bash
#SBATCH --job-name=inr2vec
#SBATCH --output=logs/%A_%x.out
#SBATCH --error=logs/%A_%x.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=universe
#SBATCH --qos=master-queuesave

set -euo pipefail

DATASET="${1:?Usage: sbatch run_inr2vec.sh <dataset> [extra args...]}"
shift
EXTRA_ARGS="${*}"

REPO_ROOT="/vol/miltank/users/hdo/ADLM"
cd "$REPO_ROOT"
mkdir -p logs

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local} dataset=$DATASET"
nvidia-smi || true

module load python/uv
uv sync
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

IS_3D=0
[[ "$DATASET" == *"3d"* ]] && IS_3D=1

if [[ $IS_3D -eq 1 ]]; then
    BATCH_SIZE=16
else
    BATCH_SIZE=128
fi

# shellcheck disable=SC2086
uv run python -m inr2vec.inr_step1.main \
    --dataset "$DATASET" \
    --image-size 224 \
    --batch-size "$BATCH_SIZE" \
    --accum-batch-size "$BATCH_SIZE" \
    --epochs 100 \
    --lr 0.001 \
    --patience 100 \
    --seed 848577 \
    --wandb \
    --wandb-project "adlm-inr2vec" \
    $EXTRA_ARGS