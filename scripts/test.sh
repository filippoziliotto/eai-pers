#!/bin/bash

# filepath: /eai-pers/scripts/evaluate.sh

python main.py \
    --config "eai_pers.yaml" \
    --mode "eval" \
    --device "cuda" \