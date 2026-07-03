#!/bin/bash
#SBATCH --job-name=hparam_search
#SBATCH --chdir=/vol/miltank/users/qyiy/ADLM
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=universe
#SBATCH --qos=master

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 \"CLI args as one quoted string\"" >&2
    exit 1
fi

CLI_ARGS="$1"
REPO_ROOT="/vol/miltank/users/qyiy/ADLM"
cd "$REPO_ROOT"
mkdir -p logs
echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local} repo=$REPO_ROOT"
echo "[$(date -Is)] cli args: $CLI_ARGS"
nvidia-smi || true

module load python/uv
uv sync
# shellcheck disable=SC2086
uv run python -m inr2vec.inr_step1.hparam_search $CLI_ARGS
