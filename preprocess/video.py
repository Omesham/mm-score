# preprocess/video.py
from pathlib import Path
from typing import List, Iterable, Tuple

import torch
from torchvision import transforms
from decord import VideoReader, cpu
from preprocess.base import BasePreprocessor

# ---- ActionCLIP helpers -------------------------------------------------
from actionclip_model import build_model, load_checkpoint   # <- repo utilities
CKPT_PATH = "actionclip_vitb16_32f.pth"                    # adjust to your file

# ─────────────────────────────────────────────────────────────────────────
class VideoPreprocessor(BasePreprocessor):
    """
    32‑frame ActionCLIP encoder → single .npy with {"ids":[], "emb":(N,D)}
    """

    def __init__(self, device: str = "cuda"):
        # 1) build empty ActionCLIP (ViT‑B/16 backbone, 32 frames)
        model = build_model(
            base_encoder      = "ViT-B/16",
            num_segments      = 32,
            tsm               = True,      # temporal shift
            drop_path_rate    = 0.0,
            pretrained_clip   = False      # we’ll load weights next
        )

        # 2) load official checkpoint
        load_checkpoint(model, CKPT_PATH, strict=True)

        super().__init__(model, device=device)

        # preprocessing – same as ActionCLIP training recipe
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),          # (C,H,W) in [0,1]
            transforms.Normalize(
                mean = [0.48145466, 0.4578275, 0.40821073],
                std  = [0.26862954, 0.26130258, 0.27577711]
            )
        ])

    # ------------------------------------------------------------------  
    def _sample_frames(self, vr: VideoReader, num: int = 32) -> torch.Tensor:
        """
        Uniformly sample `num` frames -> Tensor (T, C, H, W)
        """
        idxs = torch.linspace(0, len(vr) - 1, num).long()
        frames = vr.get_batch(idxs).permute(0, 3, 1, 2)  # T, C, H, W (RGB)
        return frames.float() / 255.0                    # [0,1] float

    def iter_samples(self, paths: List[str]) -> Iterable[Tuple[str, torch.Tensor]]:
        """
        Yield (video_id, tensor[1, T, C, H, W]) for each file.
        The outer batch dim (1) keeps BasePreprocessor unchanged.
        """
        for p in paths:
            vr = VideoReader(p, ctx=cpu(0))
            clip = self._sample_frames(vr)                      # T,C,H,W
            clip = torch.stack([self.transform(f) for f in clip])
            clip = clip.unsqueeze(0)                            # 1,T,C,H,W
            yield Path(p).stem, clip
