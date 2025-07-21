# ────────────────────────────────────────────────────────────────────────────────
# metrics/alignment.py
# Decides per‑pair whether to run temporal or semantic alignment.
#   • Temporal => TimeSformer + TimesNet encoders + cosine alignment
#   • Semantic => CLIP similarity (supports image‑text or video‑text via average frame features).
#   • Includes resumption + failed log handling like original MSCOCO-style script.
# ────────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import itertools
import numpy as np
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import clip
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from .base import Metric
from .models.timesformer import TimeSformerEncoder
from .models.timesnet import TimesNetEncoder

TEMPORAL_TYPES = {"video", "audio", "sensor"}
SAVE_PATH   = "./outputs/clip_alignment_scores.npy"
FAILED_LOG  = "./outputs/failed_log.txt"

# -----------------------------------------------------------------------------
# Helper: decide if both modalities are temporal
# -----------------------------------------------------------------------------

def _both_temporal(a: str, b: str, cfg: Dict[str, Any]) -> bool:
    def is_temp(name: str):
        spec = cfg["modalities"].get(name, {})
        return spec.get("is_temporal", spec.get("type", "").lower() in TEMPORAL_TYPES)
    return is_temp(a) and is_temp(b)

# -----------------------------------------------------------------------------
# Modular Encoder Dispatch
# -----------------------------------------------------------------------------

def get_temporal_encoder(name: str, device: str):
    if name == "video":
        return TimeSformerEncoder().to(device)
    elif name in {"sensor", "audio"}:
        return TimesNetEncoder().to(device)
    else:
        raise ValueError(f"Unsupported modality: {name}")

# -----------------------------------------------------------------------------
# Temporal Alignment: Cosine similarity of encoded sequences
# -----------------------------------------------------------------------------

def _temporal_align(seq1: torch.Tensor, seq2: torch.Tensor, mod1: str, mod2: str, device: str):
    if not isinstance(seq1, torch.Tensor) or not isinstance(seq2, torch.Tensor):
        return 0.0
    if seq1.shape[0] != seq2.shape[0]:
        min_len = min(seq1.shape[0], seq2.shape[0])
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]

    enc1 = get_temporal_encoder(mod1, device)
    enc2 = get_temporal_encoder(mod2, device)

    with torch.no_grad():
        E1 = enc1(seq1.to(device))  # (T, d)
        E2 = enc2(seq2.to(device))  # (T, d)
        sims = torch.nn.functional.cosine_similarity(E1, E2, dim=-1)  # (T,)
        score = sims.mean().item()
    return round(score, 3)

# -----------------------------------------------------------------------------
# Semantic (CLIP) alignment helper
# -----------------------------------------------------------------------------

class _ClipHelper:
    _model = None
    _pre  = None

    @classmethod
    def get(cls, device="cpu", fp16=False):
        if cls._model is None:
            model, pre = clip.load("ViT-B/32", device=device)
            model.eval()
            if fp16:
                model = model.half()
            if torch.cuda.device_count() > 1 and device.startswith("cuda"):
                print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
                model = torch.nn.DataParallel(model)
            cls._model, cls._pre = model, pre
        return cls._model, cls._pre

def _semantic_align(img_paths: List[str], caption: str, device="cpu", fp16=False):
    if not img_paths or caption is None:
        return 0.0

    model, preprocess = _ClipHelper.get(device, fp16)
    similarities = []

    try:
        images = [preprocess(Image.open(p).convert("RGB")).to(device) for p in img_paths]
        image_input = torch.stack(images)
        text_input = clip.tokenize([caption]).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=fp16):
            encoder = model.module if isinstance(model, torch.nn.DataParallel) else model
            img_feats = encoder.encode_image(image_input)
            txt_feats = encoder.encode_text(text_input)

        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
        sims = (img_feats @ txt_feats.T).squeeze().tolist()
        similarities = sims if isinstance(sims, list) else [sims]

        return round(float(np.mean(similarities)), 3)

    except Exception as e:
        with open(FAILED_LOG, "a") as f:
            f.write(f"Semantic alignment failed for {img_paths}: {str(e)}\n")
        return 0.0

# -----------------------------------------------------------------------------
# AlignmentMetric (decides temporal vs semantic)
# -----------------------------------------------------------------------------
def load_video_tensor(pt_path: str) -> torch.Tensor:
    try:
        return torch.load(pt_path)  # Assumes shape (T, C, H, W)
    except Exception as e:
        print(f"[Error] Failed loading tensor from {pt_path}: {e}")
        return torch.empty(0)


class AlignmentMetric(Metric):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16   = cfg.get("clip_model", {}).get("use_fp16", False)

    def name(self):
        return "alignment"

    def evaluate(self, modalities: Dict[str, Any]):
        out: Dict[str, Any] = {"temporal": {}, "semantic": {}}
        mods = list(modalities.keys())
        
        results = []
        start_idx = 0
        if os.path.exists(SAVE_PATH):
            results = np.load(SAVE_PATH).tolist()
            start_idx = len(results)
            print(f"Resuming from index {start_idx}")

        pair_list = list(itertools.combinations(mods, 2))

        for idx, (a, b) in enumerate(pair_list[start_idx:], start=start_idx):
            if _both_temporal(a, b, self.config):
                paths_a = modalities[a]
                paths_b = modalities[b]
                for pa, pb in tqdm(zip(paths_a, paths_b), total=min(len(paths_a), len(paths_b)), desc=f"Aligning {a} vs {b}"):
                    tensor_a = load_video_tensor(pa)
                    tensor_b = load_video_tensor(pb)
                    if tensor_a.numel() == 0 or tensor_b.numel() == 0:
                        continue
                    # reshape: (T, C, H, W) → (T, C*H*W)
                    tensor_a = tensor_a.view(tensor_a.shape[0], -1)
                    tensor_b = tensor_b.view(tensor_b.shape[0], -1)
                    sim = _temporal_align(tensor_a, tensor_b, a, b, self.device)
                    results.append(sim)
                score = round(float(np.mean(results)), 3) if results else 0.0
                out["temporal"][f"{a}_{b}_sync"] = score


            else:
                if "text" in (a, b):
                    vid_mod = b if a == "text" else a
                    text_mod = a if a == "text" else b
                    captions = modalities[text_mod][:100]
                    frames   = modalities[vid_mod][:100]
                    sims = []
                    for cap, item in tqdm(zip(captions, frames), total=min(len(captions), len(frames)), desc=f"Aligning {vid_mod} vs {text_mod}"):
                        if os.path.isfile(item):
                            img_list = [item]
                        elif os.path.isdir(item):
                            img_list = [os.path.join(item, f) for f in os.listdir(item)[:1]]
                        else:
                            with open(FAILED_LOG, "a") as f:
                                f.write(f"Missing file or directory: {item}\n")
                            continue
                        sim = _semantic_align(img_list, cap, self.device, self.fp16)
                        sims.append(sim)
                    score = round(float(np.mean(sims)), 3) if sims else 0.0
                    out["semantic"][f"{vid_mod}_{text_mod}_similarity"] = score
                    results.append(score)

            np.save(SAVE_PATH, np.array(results))

        return out
