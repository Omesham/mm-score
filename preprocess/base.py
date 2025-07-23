# preprocess/base.py
import abc, torch, numpy as np
from pathlib import Path
from typing import List, Iterable, Tuple

class BasePreprocessor(abc.ABC):
    def __init__(self, model: torch.nn.Module, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device).eval()

    # ---------- hooks -------------------------------------------------
    @abc.abstractmethod
    def iter_samples(self, paths: List[str]) -> Iterable[Tuple[str, torch.Tensor]]:
        """Yield (sample_id, prepared_tensor)"""

    @abc.abstractmethod
    def get_vector(self, batch: torch.Tensor) -> torch.Tensor:
        """Return (B, D) embeddings from model‑specific API"""

    # ---------- universal driver -------------------------------------
    @torch.inference_mode()
    def encode_and_save(self, paths: List[str], out_file: Path):
        feats, ids = [], []
        for sid, x in self.iter_samples(paths):
            vec = self.get_vector(x.to(self.device))
            feats.append(vec.cpu().numpy())
            ids.append(sid)
        np.save(out_file, {"ids": ids, "emb": np.vstack(feats)})
