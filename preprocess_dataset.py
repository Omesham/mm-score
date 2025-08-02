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
import importlib.util
import sys



# ──────────────────────────────────────────────
#  Project imports
# ──────────────────────────────────────────────
from Utils.loaders import UniversalDataLoader

# Each modality maps to its pre‑processor class  ─────────────────────────
# If a user hasn’t installed the extra deps (e.g. decord) the import
# will fail – so we try/except and mark it as “unavailable”.
# def _safe_import(path: str, cls: str):
#     try:
#         module = importlib.import_module(path)
#         return getattr(module, cls)
#     except Exception as e:         # noqa: broad‑except on purpose here
#         print(f"[Skip]   {path}.{cls} not available ({e}).")
#         return None


def _safe_import(path: str, cls: str, defer=False):
    if defer and path == "preprocess.video":
        return lambda: _safe_import(path, cls, defer=False)

    try:
        if path == "preprocess.video":
            # Force manual import from file to ensure ActionCLIP sys.path injection runs
            spec = importlib.util.spec_from_file_location(
                "preprocess.video",
                "/mmfs1/scratch/jacks.local/ojanigala/Anomaly_Detection/mm-score/preprocess/video.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules["preprocess.video"] = module
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(path)
        return getattr(module, cls)
    except Exception as e:
        print(f"[Skip]   {path}.{cls} not available ({e}).")
        return None



TextPreprocessor    = _safe_import("preprocess.text",   "TextPreprocessor")
ImagePreprocessor   = _safe_import("preprocess.image",  "ImagePreprocessor")
VideoPreprocessor = _safe_import("preprocess.video", "VideoPreprocessor", defer=True)
AudioPreprocessor   = _safe_import("preprocess.audio",  "AudioPreprocessor")
SensorPreprocessor  = _safe_import("preprocess.sensor", "SensorPreprocessor")

PREPROCESSORS = {
    "text":   TextPreprocessor,
    "image":  ImagePreprocessor,
    "video":  VideoPreprocessor,
    "audio":  AudioPreprocessor,   
    "sensor": SensorPreprocessor
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
    
        if modality == "video" and callable(preproc_cls):
            preproc_cls = preproc_cls()  # ← Load ActionCLIP only if needed
    
        if preproc_cls is None:
            print(f"[Skip] No pre‑processor implemented for “{modality}”.")
            continue
        if preproc_cls is False:   # import failed earlier
            continue       
    
        print(f"[Run]  {modality:6} | {len(paths):5} files → {out_root}/{modality}.npy")
    
        if modality == "audio":
            pre = preproc_cls(device=args.device,
                              use_clap=args.use_clap,
                              use_speecht5=args.use_speecht5)
        else:
            pre = preproc_cls(device=args.device)
    
        pre.encode_and_save(
            paths,
            out_root / f"{modality}.npy",
            max_items=args.max_items  # forward the CLI flag
        )



    print("\n  All embeddings saved in", out_root)


# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset root.")
    parser.add_argument("--out",     default="embeddings", help="Output directory.")
    parser.add_argument("--device",  default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-items", type=int,
                    help="Cap samples per modality while preprocessing")
    args = parser.parse_args()          
    max_items = args.max_items         

    main(args)
