# preprocess/clip_base.py
import clip, torch
from preprocess.base import BasePreprocessor

# load once at import time
_clip_model, _clip_preprocess = clip.load("ViT-B/32", device="cuda", jit=False)

class CLIPPreprocessor(BasePreprocessor):
    def __init__(self, device: str = "cuda"):
        super().__init__(_clip_model, device=device)

    # subclasses only implement iter_samples --------------------------
    def get_vector(self, batch: torch.Tensor) -> torch.Tensor:
        return (_clip_model.encode_image(batch)
                if batch.dtype == torch.float32        # images → float tensor
                else _clip_model.encode_text(batch))   # token ids → long tensor

# expose the transform so others can import it
clip_preprocess = _clip_preprocess
