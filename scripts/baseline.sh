#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --chdir=<path>
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --partition=<partition>
#SBATCH --qos=<qos>

set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 baseline.main CLI args as one quoted string" >&2
    exit 1
fi

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
PYTHONPATH=src uv run python -m baseline.main $CLI_ARGS