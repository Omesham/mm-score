#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────
#  preprocess/sensor.py
#
#  Purpose:
#  --------
#  • Load arbitrary multivariate time‑series (CSV / .npy / .npz …).
#  • Encode each trace with a *pre‑trained* TS2Vec backbone.
#  • Optionally project the backbone’s native dimension (D₀) to the
#    common 512‑D space used by our other modalities (CLIP, CLAP, etc.).
#  • Save **one** file  ->  embeddings/sensor.npy
#    Format: {"ids": [sample_ids], "emb": ndarray (N, 512)}
# ──────────────────────────────────────────────────────────────

from pathlib import Path
from typing import List, Iterable, Tuple

# -------- DL / math libs --------------------------------------
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# -------- project imports -------------------------------------
from preprocess.base import BasePreprocessor          # our common base class
from loaders import UniversalDataLoader               # only used by the run() helper
from ts2vec import TS2Vec                             # make sure the ts2vec package is installed

# Standardise every modality to 512‑dimensional embeddings
TARGET_DIM = 512

# ──────────────────────────────────────────────────────────────
# Projection head:  Identity  if D₀==512, else  Linear(D₀→512)
# ──────────────────────────────────────────────────────────────
class _ProjHead(nn.Module):
    """Lightweight projection so final vectors are always 512‑D."""

    def __init__(self, in_dim: int):
        super().__init__()

        # If the backbone already outputs 512, no projection is needed
        if in_dim == TARGET_DIM:
            self.proj = nn.Identity()                    # keeps data unchanged
        else:
            # Single linear layer *without* bias (bias adds no benefit here)
            self.proj = nn.Linear(in_dim, TARGET_DIM, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (B, in_dim)
        ─────────────────────────────
        Returns tensor of shape (B, 512)
        """
        return self.proj(x)


# ──────────────────────────────────────────────────────────────
#  Main pre‑processor class for sensor / time‑series data
# ──────────────────────────────────────────────────────────────
class SensorPreprocessor(BasePreprocessor):
    """
    Wrapper around a pre‑trained TS2Vec model + projection head.

    BasePreprocessor handles:
      • device placement
      • encode_and_save() loop
      • batching logic (1 sample at a time here)

    We only need to implement:
      • __init__()           → build backbone + projection
      • iter_samples()       → how to yield (sample_id, tensor) pairs
      • get_vector() (opt.)  → custom forward pass if backbone needs it
    """

    def __init__(self, device: str = "cuda"):
        # 1) Load pre‑trained TS2Vec backbone
        backbone = TS2Vec.load_pretrained("ts2vec_multivar")
        backbone.eval()                                        # inference‑only

        # 2) Figure out its native output dimension D₀
        with torch.inference_mode():
            dummy = torch.randn(1, 5, 128)                     # fake (B, C, T)
            native_dim = backbone.encode(dummy).shape[-1]

        # 3)  Wrap backbone + projection head in nn.Sequential
        model_with_proj = nn.Sequential(
            backbone,                                          # (B, C, T) → (B, D₀)
            _ProjHead(native_dim)                              # (B, D₀) → (B, 512)
        )

        # 4)  Pass to BasePreprocessor for device handling, etc.
        super().__init__(model_with_proj, device=device)

    # ──────────────────────────────────────────────────────────
    #  Helpers to turn *files* into float32 tensors
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _load_trace(fp: str) -> torch.Tensor:
        """
        Reads a single time‑series file and returns a FloatTensor (C, T).

        Accepted formats:
          • .npy / .npz : NumPy arrays saved previously
          • .csv / .txt : plain text files   (no header)
        """
        ext = Path(fp).suffix.lower()

        # ------------ load into NumPy -------------------------
        if ext in {".npy", ".npz"}:
            arr = np.load(fp)
        else:
            arr = pd.read_csv(fp, header=None).values

        # Convert to float32 Tensor, transpose so shape = (C, T)
        return torch.from_numpy(arr.astype("float32")).t()

    # -----------------------------------------------------------------
    #   Yield samples one‑by‑one for encode_and_save()
    # -----------------------------------------------------------------
    def iter_samples(self, paths: List[str]) -> Iterable[Tuple[str, torch.Tensor]]:
        """
        For each file path:
            •  load & preprocess → tensor (C, T)
            •  add batch dim → (1, C, T)
            •  yield (sample_id, tensor)
        """
        for p in paths:
            yield Path(p).stem, self._load_trace(p).unsqueeze(0)

    # -----------------------------------------------------------------
    #   Custom forward pass (optional)
    # -----------------------------------------------------------------
    def get_vector(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Overrides BasePreprocessor.get_vector() so we can call
        our Sequential(model+proj) directly.
        """
        return self.model(batch.to(self.device))


# ──────────────────────────────────────────────────────────────
#  Convenience helper for quick, stand‑alone testing
# ──────────────────────────────────────────────────────────────
def run(dataset_root: str, out_dir: str, device="cuda"):
    """
    Usage (from repo root):

        python -m preprocess.sensor run /path/to/data ./embeddings --device cpu

    * Finds all sensor‑type files under `dataset_root`
    * Encodes them → one   embeddings/sensor.npy
    """

    detected = UniversalDataLoader.auto_detect_modalities(dataset_root)
    sensor_files = detected.get("sensor", [])
    if not sensor_files:
        print("[SensorPreproc] No sensor files found.")
        return

    pre = SensorPreprocessor(device=device)
    out_path = Path(out_dir) / "sensor.npy"
    pre.encode_and_save(sensor_files, out_path)
    print(f"✅  Saved: {out_path}")

