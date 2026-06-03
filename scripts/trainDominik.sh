#!/bin/bash
#SBATCH --job-name=gnn-chestmnist
#SBATCH --output=/vol/miltank/users/hdo/logs/%A.out
#SBATCH --error=/vol/miltank/users/hdo/logs/%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=master-queuesave

ml python/uv

mkdir -p /vol/miltank/users/hdo/logs

cd /vol/miltank/users/hdo/ADLM

uv run python train_gnn.py \
    --dataset chestmnist \
    --data-root /vol/miltank/users/hdo/data \
    --max-samples 500 \
    --epochs 50 \
    --batch-size 64 \
    --num-workers 4