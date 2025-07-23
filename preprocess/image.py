# preprocess/image.py  (CLIP version)
from pathlib import Path
from typing import List, Iterable, Tuple
import torch
from PIL import Image
import clip                               # ← OpenAI CLIP

from preprocess.base import BasePreprocessor
from loaders import UniversalDataLoader   # tie‑in for the run() helper

# ──────────────────────────────────────────────
# Load CLIP model & preprocessing once
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")  # set device in __init__

class ImagePreprocessor(BasePreprocessor):
    """
    Encode images with CLIP (ViT‑B/32) → one image.npy containing:
        {"ids": [...], "emb": (N, 512) float32}
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(clip_model, device=device)  # store model on device

    # ---------- Loader hook ----------------------------------
    def iter_samples(self, paths: List[str]) -> Iterable[Tuple[str, torch.Tensor]]:
        """Yield (id, tensor) for every image file."""
        for p in paths:
            img = Image.open(p).convert("RGB")
            tensor = clip_preprocess(img)            # (3, 224, 224)
            yield Path(p).stem, tensor.unsqueeze(0)  # add batch dim


# ──────────────────────────────────────────────
# Helper to run stand‑alone
def run(dataset_root: str, out_dir: str, device="cuda"):
    detected = UniversalDataLoader.auto_detect_modalities(dataset_root)
    image_files = detected.get("image", [])
    if not image_files:
        print("[ImagePreproc] No images found.")
        return
    pre = ImagePreprocessor(device=device)
    pre.encode_and_save(image_files, Path(out_dir) / "image.npy")
