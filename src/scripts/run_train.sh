#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

SESSION_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logger/session_${SESSION_ID}_FINAL"
mkdir -p "$LOG_DIR"

echo "🚀 Starting Master Session: ${SESSION_ID}"

export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4 
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,5

DATASETS=("politifact" "gossipcop" "fakeddit")

for ds in "${DATASETS[@]}"
do
    echo "🔥 Current Job: ${ds^^}"

    accelerate launch \
        --num_processes 5 \
        --mixed_precision fp16 \
        src/train.py \
        --dataset_name "$ds" \
        --output_dir ./checkpoints \
        --batch_size 1 \
        --grad_accum_steps 16 \
        --epochs 50 \
        --patience 5 \
        --lr_lora 2e-5 \
        --lr_head 5e-5 \
        --lora_r 16 \
        --lora_alpha 32 \
        --aux_weight 0.3 \
        --num_workers 0 \
        --seed 3407 \
        > "$LOG_DIR/train_${ds}.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ Completed: ${ds^^}"
    else
        echo "❌ Failed: ${ds^^}"
    fi

    echo "💤 Cooling down..."
    sleep 60
done

echo "🎉 All specialist models finished!"