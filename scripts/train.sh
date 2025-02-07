#!/bin/bash

# filepath: /eai-pers/scripts/evaluate.sh

python main.py \
    --data_dir "data" \
    --data_split "val" \
    --mode "train" \
    --seed 2025 \
    --batch_size 1 \
    --num_workers 4 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --num_epochs 2 \
    --loss_choice "L2" \
    --embed_dim 768 \
    --num_heads 6 \
    --freeze_encoder \
    --map_size 500 \
    --pixels_per_meter 10 \
    --optimizer "adam" \
    --scheduler "none" \
    --step_size 5 \
    --gamma 0.1 \
    --patience 10 \
    --device "mps" \
    #--use_wandb \
    #--run_name "prova-train"