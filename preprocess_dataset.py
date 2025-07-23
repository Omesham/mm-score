#!/usr/bin/env python3
"""
Batch‑preprocess an entire multimodal dataset and write
**one .npy per modality** (image.npy, video.npy, …) into <out_dir>.

Example:
    python preprocess_dataset.py \
           --dataset /data/my_dataset \
           --out     embeddings \
           --device  cuda
"""
from pathlib import Path
import argparse
import importlib

# ──────────────────────────────────────────────
#  Project imports
# ──────────────────────────────────────────────
from loaders import UniversalDataLoader

# Each modality maps to its pre‑processor class  ─────────────────────────
# If a user hasn’t installed the extra deps (e.g. decord) the import
# will fail – so we try/except and mark it as “unavailable”.
def _safe_import(path: str, cls: str):
    try:
        module = importlib.import_module(path)
        return getattr(module, cls)
    except Exception as e:         # noqa: broad‑except on purpose here
        print(f"[Skip]   {path}.{cls} not available ({e}).")
        return None

ImagePreprocessor   = _safe_import("preprocess.image",  "ImagePreprocessor")
VideoPreprocessor   = _safe_import("preprocess.video",  "VideoPreprocessor")
AudioPreprocessor   = _safe_import("preprocess.audio",  "AudioPreprocessor")
TextPreprocessor    = _safe_import("preprocess.text",   "TextPreprocessor")
SensorPreprocessor  = _safe_import("preprocess.sensor", "SensorPreprocessor")

PREPROCESSORS = {
    "image":  ImagePreprocessor,
    "video":  VideoPreprocessor,
    "audio":  AudioPreprocessor,
    "text":   TextPreprocessor,
    "sensor": SensorPreprocessor,
}
# ──────────────────────────────────────────────


def main(args):
    dataset_root = Path(args.dataset).resolve()
    out_root     = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Auto‑detect all modality files once
    detected = UniversalDataLoader.auto_detect_modalities(str(dataset_root))

    # 2) Loop through what we found
    for modality, paths in detected.items():
        preproc_cls = PREPROCESSORS.get(modality)
        if preproc_cls is None:
            print(f"[Skip] No pre‑processor implemented for “{modality}”.")
            continue
        if preproc_cls is False:   # import failed earlier
            continue

        print(f"[Run]  {modality:6} | {len(paths):5} files → {out_root}/{modality}.npy")
        pre = preproc_cls(device=args.device)
        pre.encode_and_save(paths, out_root / f"{modality}.npy")

    print("\n  All embeddings saved in", out_root)


# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset root.")
    parser.add_argument("--out",     default="embeddings", help="Output directory.")
    parser.add_argument("--device",  default="cuda", choices=["cuda", "cpu"])
    main(parser.parse_args())
