# preprocess/image.py  (CLIP version)
from pathlib import Path
from typing import List, Iterable, Tuple
import torch
from PIL import Image
import clip

from preprocess.base import BasePreprocessor
from Utils.loaders import UniversalDataLoader

# ------------------------------------------------------------
_clip, _preproc = clip.load("ViT-B/32", device="cpu")  # stays global

class ImagePreprocessor(BasePreprocessor):
    """CLIP‑ViT‑B/32 image encoder -> one image.npy (ids, emb 512‑D)."""

    def __init__(self, device: str = "cuda"):
        super().__init__(_clip, device=device)          # model moved to GPU

    # mandatory for BasePreprocessor
    def get_vector(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(batch)           # (B,512)

    def iter_samples(self, paths: List[str]) -> Iterable[Tuple[str, torch.Tensor]]:
        for p in paths:
            img = Image.open(p).convert("RGB")
            yield Path(p).stem, _preproc(img).unsqueeze(0)

# ------------------------------------------------------------
def run(dataset_root: str, out_dir: str, device="cuda"):
    det = UniversalDataLoader.auto_detect_modalities(dataset_root)
    imgs = det.get("image", [])
    if not imgs:
        print("[ImagePreproc] No images found."); return
    ImagePreprocessor(device).encode_and_save(imgs, Path(out_dir) / "image.npy")
