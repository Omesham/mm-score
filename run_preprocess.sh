#!/usr/bin/env bash
# Make executable:  chmod +x run_preprocess.sh

python preprocess_dataset.py \
    --dataset /data/my_dataset \
    --out embeddings \
    --device cuda \
    --max-items 10 \
    --use_clap \
    --use_speecht5
