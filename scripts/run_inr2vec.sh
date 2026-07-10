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

INR_ROOT="/vol/miltank/projects/practical_sose26/gaussian/data/inr2vec/$DATASET"

echo "[$(date -Is)] Step 1: fitting INRs for $DATASET"
uv run python -m inr2vec.inr_step1.pretrain \
    --dataset "$DATASET" \
    --splits train val test \
    --output-dir "/vol/miltank/projects/practical_sose26/gaussian/data" \
    --epochs 2000 \
    --lr 0.001 \
    --patience 100 \
    --image-size 224

echo "[$(date -Is)] Step 2: training inr2vec_paper on $DATASET"
# shellcheck disable=SC2086
uv run python -m baseline.main \
    --model inr2vec_paper \
    --dataset "$DATASET" \
    --image-size 224 \
    --inr-root "$INR_ROOT" \
    --batch-size "$BATCH_SIZE" \
    --accum-batch-size "$BATCH_SIZE" \
    --epochs 100 \
    --lr 0.001 \
    --patience 100 \
    --seed 848577 \
    --num-workers 2 \
    --wandb \
    --wandb-project "adlm-inr2vec" \
    $EXTRA_ARGS