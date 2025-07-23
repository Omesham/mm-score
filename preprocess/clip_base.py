# preprocess/clip_base.py
import clip, torch
from preprocess.base import BasePreprocessor

class CLIPPreprocessor(BasePreprocessor):
    def __init__(self, device="cuda"):
        self.clip, _ = clip.load("ViT-B/32", device=device, jit=False)
        super().__init__(self.clip, device=device)

    # subclasses only implement iter_samples --------------------------
    def get_vector(self, batch: torch.Tensor) -> torch.Tensor:
        # decide on tower by dtype → image = float / text = long
        return (self.clip.encode_image(batch)
                if batch.dtype == torch.float32
                else self.clip.encode_text(batch))
