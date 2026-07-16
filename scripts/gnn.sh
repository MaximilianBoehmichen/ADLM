#!/bin/bash
#SBATCH --job-name=gnn-test
#SBATCH --chdir=<path>
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --partition=<partition>
#SBATCH --qos=<qos>

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 baseline.main CLI args as one quoted string" >&2
    exit 1
fi

export UV_CACHE_DIR="<path>"
CLI_ARGS="$1"
REPO_ROOT="<path>"
cd "$REPO_ROOT"
mkdir -p logs
echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local} repo=$REPO_ROOT"
echo "[$(date -Is)] cli args: $CLI_ARGS"
nvidia-smi || true

module load python/uv
uv sync
# shellcheck disable=SC2086  # intentional word-splitting of the string
uv run python train_gnn.py --dataset "chestmnist" --num-workers 4 --wandb-project "adlm-gnn-experiment" --epochs 100 --in-memory --data-root "<data-root>" $CLI_ARGS
uv run python train_gnn.py --dataset "organcmnist" --num-workers 4 --wandb-project "adlm-gnn-experiment" --epochs 100 --in-memory --data-root "<data-root>" $CLI_ARGS
uv run python train_gnn.py --dataset "organmnist3d" --batch-size 16 --num-workers 4 --wandb-project "adlm-gnn-experiment" --epochs 100 --in-memory --data-root "<data-root>" $CLI_ARGS
uv run python train_gnn.py --dataset "fracturemnist3d" --batch-size 16 --num-workers 4 --wandb-project "adlm-gnn-experiment" --epochs 100 --in-memory --data-root "<data-root>" $CLI_ARGS