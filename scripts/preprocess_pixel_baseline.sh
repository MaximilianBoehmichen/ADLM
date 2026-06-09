#!/bin/bash
#SBATCH --job-name=pixel-preprocess
#SBATCH --output=/vol/miltank/users/hdo/logs/%A.out
#SBATCH --error=/vol/miltank/users/hdo/logs/%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=master-queuesave

source /usr/share/modules/init/bash
ml python/uv

mkdir -p /vol/miltank/users/hdo/logs

cd /vol/miltank/users/hdo/ADLM

uv run python preprocess_pixel_baseline.py \
    --dataset chestmnist \
    --splits train val test \
    --output-dir /vol/miltank/users/hdo/data_pixel \
    --stride 7 \
    --k-graph 15