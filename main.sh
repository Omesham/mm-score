#!/usr/bin/env bash
# ------------------------------------------------------------------
# Run MM‑Score with a fixed dataset and settings
# ------------------------------------------------------------------

python mm_score_main.py \
  --dataset /data/mm_demo \
  --modalities image audio \
  --max-items 500 \
  --preprocess \
  --output demo_results.json

