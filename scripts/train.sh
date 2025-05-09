#!/bin/bash

# filepath: /eai-pers/scripts/evaluate.sh

python main.py \
    --config "default.yaml" \
    --data_dir "data" \
    --data_split "object_unseen" \
    --mode "train" \
    --increase_dataset_size \
    --validate_after_n_epochs 1 \
    --seed 2025 \
    --batch_size 8 \
    --num_workers 4 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --num_epochs 2 \
    --loss_choice "CE" \
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
    --save_checkpoint \
    --checkpoint_path "model/checkpoints/model.pth" \
    --use_aug \
    --use_horizontal_flip \
    --use_vertical_flip \
    --use_random_rotation \
    --use_desc_aug \
    --aug_prob 0.5 \
    #--use_wandb \
    #--run_name "prova-train"