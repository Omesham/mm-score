# preprocess/video.py
"""
ActionCLIP‑based video encoder
Saves ONE   video.npy   with  {"ids": [...], "emb": (N, 512)}
"""

from pathlib import Path
from typing  import List, Iterable, Tuple
import torch, torchvision
import decord                          # pip install decord
import clip                            # repo provides ActionCLIP wrapper too

from preprocess.base       import BasePreprocessor
from preprocess.clip_base  import clip_preprocess          # reuse resize / crop
from loaders               import UniversalDataLoader      # used only by run()

# ---------------------------------------------------------------------------
# 1) load ActionCLIP *once* at import time
actionclip, _ = clip.load(
    model_name      = "ViT-B/16",
    device          = "cpu",          # moved to GPU in __init__
    tsm             = True,           # enable temporal shift
    T               = 32,             # frames per clip
    jit             = False)

# weights (download separately – 335 MB) --------------------------
#   wget https://link.to/actionclip_vit_b_32.pth -O weights/actionclip_vit_b_32.pth
CKPT = Path("weights/actionclip_vit_b_32.pth")
state_dict = torch.load(CKPT, map_location="cpu")
actionclip.load_state_dict(state_dict.get("state_dict", state_dict), strict=False)
actionclip.eval()

# ---------------------------------------------------------------------------
class VideoPreprocessor(BasePreprocessor):
    """
    1. uniformly sample 32 frames per file
    2. feed to ActionCLIP → 512‑D global embedding
    3. store everything in one .npy
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(actionclip, device=device)
        self.resize = torchvision.transforms.Resize(224)

    # -------------- helpers -------------------------------------------------
    def _sample_frames(self, vr, T=32):
        idxs = torch.linspace(0, len(vr) - 1, T).long()
        return vr.get_batch(idxs)          # (T, H, W, 3)  uint8

    # -------------- Base hook ----------------------------------------------
    def iter_samples(self, paths: List[str]) -> Iterable[Tuple[str, torch.Tensor]]:
        for p in paths:
            vr   = decord.VideoReader(p, width=256, height=256)
            clip = self._sample_frames(vr)                         # (T,H,W,3)

            # → (T,3,224,224) float32 in [0,1] with CLIP transform
            clip = torch.stack([clip_preprocess(Image=self.resize(frame)) for frame in clip])
            yield Path(p).stem, clip.unsqueeze(0)                  # add batch‑dim


# ---------------------------------------------------------------------------
def run(dataset_root: str, out_dir: str, device="cuda"):
    """Quick CLI / notebook test: encode ONLY the videos under dataset_root."""
    detected = UniversalDataLoader.auto_detect_modalities(dataset_root)
    video_files = detected.get("video", [])
    if not video_files:
        print("[VideoPreproc] No videos found.")
        return
    pre = VideoPreprocessor(device=device)
    pre.encode_and_save(video_files, Path(out_dir) / "video.npy")
# ---------------------------------------------------------------------------
