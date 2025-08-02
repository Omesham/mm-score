#!/usr/bin/env bash

python mm_score_main.py \
  --config config_mm_score.yaml \
  --dataset /mmfs1/scratch/jacks.local/ojanigala/Anomaly_Detection/mm-score/MSCOCO \
  --preprocess \
  --max-items 10 \
  --output results_200.json

