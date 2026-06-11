#!/bin/bash
#SBATCH --job-name=linear-baseline
#SBATCH --output=/vol/miltank/users/hdo/logs/%A.out
#SBATCH --error=/vol/miltank/users/hdo/logs/%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=master-queuesave

ml python/uv

mkdir -p /vol/miltank/users/hdo/logs

cd /vol/miltank/users/hdo/ADLM

uv run python scripts/linear_baseline.py \
    --data-root /vol/miltank/users/hdo/data \
    --dataset chestmnist