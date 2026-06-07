#!/bin/bash
# Submit one job per layer count (0-3) to find where message passing helps.
# Usage: bash scripts/layerSweep.sh

for LAYERS in 0 1 2 3; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gnn-layers${LAYERS}
#SBATCH --output=/vol/miltank/users/hdo/logs/%A-layers${LAYERS}.out
#SBATCH --error=/vol/miltank/users/hdo/logs/%A-layers${LAYERS}.err
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
    --epochs 50 \
    --batch-size 256 \
    --num-workers 4 \
    --layers ${LAYERS} \
    --hidden 64 \
    --weight-decay 1e-4
EOF
    echo "Submitted layers=${LAYERS}"
done