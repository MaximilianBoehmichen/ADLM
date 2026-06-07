#!/bin/bash
# Sweep across model complexity levels. Respects 2-job limit.
# Usage: bash scripts/modelSweep.sh [MODELS...]
# Default: mlp gcn (first 2). Run "relpos full" after those finish.

MODELS="${@:-mlp gcn}"

for MODEL in $MODELS; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gnn-${MODEL}
#SBATCH --output=/vol/miltank/users/hdo/logs/%A-${MODEL}.out
#SBATCH --error=/vol/miltank/users/hdo/logs/%A-${MODEL}.err
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
    --model ${MODEL} \
    --dataset chestmnist \
    --data-root /vol/miltank/users/hdo/data \
    --epochs 50 \
    --batch-size 256 \
    --num-workers 4 \
    --layers 2 \
    --hidden 64 \
    --weight-decay 1e-3 \
    --dropout 0.3
EOF
    echo "Submitted model=${MODEL}"
done
