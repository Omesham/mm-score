# preprocess/image.py  (CLIP version)
from pathlib import Path
from typing import List, Iterable, Tuple
import torch
from PIL import Image
import clip
from typing import Optional
from preprocess.base import BasePreprocessor
from Utils.loaders import UniversalDataLoader

# ------------------------------------------------------------
# _clip, _preproc = clip.load("ViT-B/32", device="cpu")  # stays global
def load_clip_model(device):
    return clip.load("ViT-B/32", device=device)


_clip, _preproc = load_clip_model("cuda")

class ImagePreprocessor(BasePreprocessor):
    """CLIP‑ViT‑B/32 image encoder -> one image.npy (ids, emb 512‑D)."""

    def __init__(self, device: str = "cuda"):
        super().__init__(_clip, device=device)          # model moved to GPU

    # mandatory for BasePreprocessor
    def get_vector(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(batch)           # (B,512)

    def iter_samples(self, paths: List[str], max_items: Optional[int] = None) -> Iterable[Tuple[str, torch.Tensor]]:
        count = 0  # ← Initialize counter
        print(f"Printing count: {count}")
        for p in paths:
            if max_items is not None and count >= max_items:
                break
            # print("Printing path:", p)
            img = Image.open(p).convert("RGB")
            print(f"Printing count: {count}")
            yield Path(p).stem, _preproc(img).unsqueeze(0)
            count += 1  # ← Increment after yielding
# ------------------------------------------------------------
def run(dataset_root: str, out_dir: str, device="cuda"):
    det = UniversalDataLoader.auto_detect_modalities(dataset_root)
    imgs = det.get("image", [])
    if not imgs:
        print("[ImagePreproc] No images found."); return
    ImagePreprocessor(device).encode_and_save(imgs, Path(out_dir) / "image.npy")
