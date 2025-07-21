
#!/usr/bin/env bash
# Make executable:  chmod +x run_preprocess.sh

python preprocess_dataset.py \
  --dataset /path/to/your/dataset \
  --out     embeddings \
  --device  cuda
