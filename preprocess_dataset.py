#!/usr/bin/env python3
"""
Batch-preprocess an entire multimodal dataset and save ONE .npy per modality.

Usage:
    python preprocess_dataset.py --dataset /data/my_dataset \
                                 --out     embeddings \
                                 --device  cuda
"""
import argparse
from pathlib import Path

# --- MM-SCORE imports -------------------------------------------------
from loaders import UniversalDataLoader
from preprocess.image  import ImagePreprocessor
from preprocess.text import TextPreprocessor

# Map modality names → preprocessor classes
PREPROCESSORS = {
    "image":  ImagePreprocessor,
    "video":  VideoPreprocessor,
    "audio":  AudioPreprocessor,
    "text":   TextPreprocessor,
    "sensor": SensorPreprocessor
}

# ---------------------------------------------------------------------
def main(args):
    dataset_root = Path(args.dataset).resolve()
    out_root     = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Find all modality files once
    detected = UniversalDataLoader.auto_detect_modalities(str(dataset_root))

    # 2) Loop through supported modalities
    for modality, paths in detected.items():
        if modality not in PREPROCESSORS:
            print(f"[Skip] No preprocessor implemented for '{modality}'.")
            continue

        print(f"[Run] {modality}: {len(paths)} files → {out_root}/{modality}.npy")
        pre = PREPROCESSORS[modality](device=args.device)
        pre.encode_and_save(paths, out_root / f"{modality}.npy")

    print("\n✅  Pre-processing finished.")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to dataset root.")
    ap.add_argument("--out",     default="embeddings", help="Output dir.")
    ap.add_argument("--device",  default="cuda", choices=["cuda", "cpu"])
    main(ap.parse_args())

