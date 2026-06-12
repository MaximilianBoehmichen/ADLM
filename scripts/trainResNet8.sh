#!/bin/bash
#SBATCH --job-name=gnn-resnet8
#SBATCH --output=/vol/miltank/users/hdo/logs/%A.out
#SBATCH --error=/vol/miltank/users/hdo/logs/%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=master-queuesave

source /usr/share/modules/init/bash
ml python/uv

mkdir -p /vol/miltank/users/hdo/logs

cd /vol/miltank/users/hdo/ADLM

uv run python train_gnn.py \
    --dataset organcmnist \
    --data-root /vol/miltank/users/hdo/data \
    --epochs 100 \
    --batch-size 256 \
    --num-workers 4 \
    --layers 6 \
    --hidden 112 \
    --max-samples 5000
    