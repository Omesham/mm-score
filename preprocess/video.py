# preprocess/video.py
"""
ActionCLIP‑based video encoder
Saves ONE   video.npy   with  {"ids": [...], "emb": (N, 512)}
"""
# ----------------------------------------------------------------------
from typing import Optional
from pathlib import Path
from typing import List, Iterable, Tuple, Optional
import torch
import torchvision
import decord                          # pip install decord
from PIL import Image
from preprocess.base import BasePreprocessor
from preprocess.clip_base import clip_preprocess          # reuse resize / crop
from Utils.loaders import UniversalDataLoader      # used only by run()
import sys
sys.path.insert(0, "/mmfs1/scratch/jacks.local/ojanigala/ActionCLIP")
sys.modules.pop("clip", None)
import clip
# -------------------- Config -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = Path("/mmfs1/scratch/jacks.local/ojanigala/ActionCLIP/vit-b-16-32f.pt")

# Load ActionCLIP model structure
actionclip, _ = clip.load("ViT-B/16", device=device, tsm=True, T=32, jit=False)

# Load pretrained weights from checkpoint directly to device
state_dict = torch.load(CKPT_PATH, map_location=device)

# Handle checkpoints that have 'state_dict' wrapper
state_dict = state_dict.get("state_dict", state_dict)

# Load weights into model
actionclip.load_state_dict(state_dict, strict=False)

# Move model to device and set to eval mode
actionclip.to(device).eval()
# ------------------------------------------------

class VideoPreprocessor(BasePreprocessor):
    """
    1. Uniformly sample 32 frames per video file.
    2. Feed to ActionCLIP → 512‑D global embedding.
    3. Store everything in one .npy file.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(actionclip, device=device)

    # -------- Frame Sampling Helper -------------
    def _sample_frames(self, vr, T=32):
        idxs = torch.linspace(0, len(vr) - 1, T).long()
        return vr.get_batch(idxs)  # (T, H, W, 3) uint8

    # -------- Main Sample Generator ------------
    def iter_samples(self, paths: List[str], max_items: Optional[int] = None) -> Iterable[Tuple[str, torch.Tensor]]:
        count = 0
        for p in paths:
            if max_items is not None and count >= max_items:
                break
            vr = decord.VideoReader(p, width=256, height=256)
            clip_frames = self._sample_frames(vr)  # (T,H,W,3)

            tensors = []
            for frame in clip_frames:
                pil = Image.fromarray(frame.asnumpy())
                tensors.append(clip_preprocess(pil))  # (3,224,224)

            clip_tensor = torch.stack(tensors)  # (T,3,224,224)
            yield Path(p).stem, clip_tensor.unsqueeze(0)  # (1,T,3,224,224)
            count += 1

# ------------------------------------------------
def run(dataset_root: str, out_dir: str, device="cuda"):
    """Quick CLI / notebook test: encode ONLY the videos under dataset_root."""
    detected = UniversalDataLoader.auto_detect_modalities(dataset_root)
    video_files = detected.get("video", [])
    if not video_files:
        print("[VideoPreproc] No videos found.")
        return
    pre = VideoPreprocessor(device=device)
    pre.encode_and_save(video_files, Path(out_dir) / "video.npy")
# ------------------------------------------------
