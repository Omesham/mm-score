# preprocess/base.py
import abc, torch, numpy as np
from pathlib import Path
from typing import List, Iterable, Tuple
from typing import Optional


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
    def encode_and_save(
        self,
        paths: List[str],
        out_file: Path,
        max_items: Optional[int] = None,
        batch_size: int = 64           # ← you can expose this as a CLI flag later
    ):
        all_feats, all_ids = [], []
        cache_x, cache_id  = [], []
    
        for sid, x in self.iter_samples(paths, max_items=max_items):
            cache_x.append(x)          # (1, C, H, W) or (1, 77) etc.
            cache_id.append(sid)
    
            # flush when batch full
            if len(cache_x) == batch_size:
                batch = torch.cat(cache_x).to(self.device, non_blocking=True)
                feats = self.get_vector(batch).cpu().numpy()    # (B,512)
                all_feats.append(feats)
                all_ids.extend(cache_id)
                cache_x, cache_id = [], []
    
        # flush last partial batch
        if cache_x:
            batch = torch.cat(cache_x).to(self.device, non_blocking=True)
            feats = self.get_vector(batch).cpu().numpy()
            all_feats.append(feats)
            all_ids.extend(cache_id)
    
        out_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_file, {"ids": all_ids, "emb": np.vstack(all_feats)})

