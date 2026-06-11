#!/bin/bash
#SBATCH --job-name=preprocess-organcmnist
#SBATCH --output=/vol/miltank/users/hdo/logs/%A.out
#SBATCH --error=/vol/miltank/users/hdo/logs/%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=master-queuesave

ml python/uv

mkdir -p /vol/miltank/users/hdo/logs

cd /vol/miltank/users/hdo/ADLM

uv run python preprocess_dataset.py \
    --dataset organcmnist \
    --splits train val test \
    --output-dir /vol/miltank/projects/practical_sose26/gaussian/data/organcmnist \
    --medmnist-root /vol/miltank/users/hdo/medmnist