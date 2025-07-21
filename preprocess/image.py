
from pathlib import Path
from typing import List, Iterable, Tuple

import torch
from PIL import Image
from torchvision import models, transforms

from preprocess.base import BasePreprocessor
from loaders import UniversalDataLoader   # <-- tie-in

# ────────────────────────────────────────────────────────────
_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                # (C,H,W) in [0,1]
])

class ImagePreprocessor(BasePreprocessor):
    """
    Encode images once → .npy cache.
    Accepts a *list of file paths* produced by UniversalDataLoader.
    """

    def __init__(self, device: str = "cuda"):
        backbone = models.resnet50(weights="DEFAULT")
        backbone.fc = torch.nn.Identity()              # 2048-D global feature
        super().__init__(backbone, device=device)

    # ---------- Loader hook ----------------------------------
    def iter_samples(self, paths: List[str]) -> Iterable[Tuple[str, torch.Tensor]]:
        """Yield (id, tensor) for every image file in list `paths`."""
        for p in paths:
            img = Image.open(p).convert("RGB")
            yield Path(p).stem, _tf(img).unsqueeze(0)   # add batch dim


# ────────────────────────────────────────────────────────────
# helper to run end-to-end once
def run(dataset_root: str, out_dir: str, device="cuda"):
    detected = UniversalDataLoader.auto_detect_modalities(dataset_root)
    image_files = detected.get("image", [])
    if not image_files:
        print("[ImagePreproc] No images found.")
        return
    pre = ImagePreprocessor(device=device)
    pre.encode_and_save(image_files, Path(out_dir) / "image")
