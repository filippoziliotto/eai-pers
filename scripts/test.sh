#!/bin/bash

# filepath: /eai-pers/scripts/evaluate.sh

python main.py \
    --data_dir "data" \
    --data_split "val" \
    --mode "eval" \
    --seed 2025 \
    --batch_size 2 \
    --num_workers 8 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --num_epochs 10 \
    --loss_choice "L2" \
    --embed_dim 512 \
    --num_heads 8 \
    --freeze_encoder \
    --map_size 500 \
    --pixels_per_meter 10 \
    --optimizer "adam" \
    --scheduler "step_lr" \
    --step_size 5 \
    --gamma 0.1 \
    --patience 10 \
    --device "cpu" \
    --use_wandb \
    --run_name "EAI-Pers-Eval"