# preprocess/video.py
from pathlib import Path
from typing import List, Iterable, Tuple
import torch, torchvision
import decord                         # pip install decord
from actionclip_model import build_model, load_checkpoint  # repo helper

from preprocess.base import BasePreprocessor

class VideoPreprocessor(BasePreprocessor):
    def __init__(self, device="cuda"):
        # 32-frame ActionCLIP (shell + weights)
        model, _ = clip.load("ViT-B/16", tsm=True, T=32, device=device, jit=False)
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device)
                              .get("state_dict", torch.load(CKPT_PATH, map_location=device)),
                              strict=False)
        model.eval()
        load_checkpoint(model, "actionclip_vit_b_32.pth")
        super().__init__(model, device=device)

        self.resize = torchvision.transforms.Resize(224)
        self.to_tensor = torchvision.transforms.ToTensor()

    def _sample_frames(self, vr, num=16):
        idxs = torch.linspace(0, len(vr)-1, num).long()
        return vr.get_batch(idxs).permute(0,3,1,2)  # T,C,H,W

    def iter_samples(self, paths: List[str]) -> Iterable[Tuple[str, torch.Tensor]]:
        for p in paths:
            vr = decord.VideoReader(p, width=256, height=256)
            clip = self._sample_frames(vr)                # (T,C,H,W)
            clip = torch.stack([self.to_tensor(self.resize(f)) for f in clip])
            clip = clip.unsqueeze(0)                     # add batch
            yield Path(p).stem, clip

