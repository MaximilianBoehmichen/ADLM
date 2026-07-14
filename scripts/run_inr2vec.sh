#!/bin/bash
#SBATCH --job-name=inr2vec
#SBATCH --chdir=/vol/miltank/users/hdo/ADLM
#SBATCH --output=logs/%A_%x.out
#SBATCH --error=logs/%A_%x.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=universe
#SBATCH --qos=master-queuesave

set -euo pipefail

REPO_ROOT="/vol/miltank/users/hdo/ADLM"
mkdir -p "$REPO_ROOT/logs"
echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi || true

module load python/uv
uv sync
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

uv run python -m baseline.main \
    --model inr2vec_input \
    --dataset chestmnist \
    --inr-root /vol/miltank/projects/practical_sose26/gaussian/data/inr2vec/chestmnist \
    --batch-size 128 \
    --wandb \
    --wandb-project ADLM-inr2vec