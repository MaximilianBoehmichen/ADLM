#!/bin/bash
# Submit one job per layer count to find where message passing helps.
# Usage: bash scripts/layerSweep.sh [LAYERS...]
# Default: layers 0 1 (2-job limit). Run again with "2 3" after first pair finishes.
# Example: bash scripts/layerSweep.sh 2 3

LAYER_LIST="${@:-0 1}"

for LAYERS in $LAYER_LIST; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gnn-layers${LAYERS}
#SBATCH --output=/vol/miltank/users/hdo/logs/%A-layers${LAYERS}.out
#SBATCH --error=/vol/miltank/users/hdo/logs/%A-layers${LAYERS}.err
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-02:00:00
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
    --epochs 50 \
    --batch-size 256 \
    --num-workers 4 \
    --layers ${LAYERS} \
    --hidden 64 \
    --weight-decay 1e-3 \
    --dropout 0.3
EOF
    echo "Submitted layers=${LAYERS}"
done