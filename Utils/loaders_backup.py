# loaders.py
# Dynamically loads data for each modality defined in the YAML configuration.
# Supports: text, image, video (frame‑folders), audio, and generic sensor files.
# ────────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import json
from typing import Dict, List, Any

import numpy as np

# Optional: only import heavy libs when needed
try:
    import soundfile as sf  # for audio duration check if available
except ImportError:
    sf = None  # type: ignore


# -----------------------------------------------------------------------------
# Helper loaders per modality type
# -----------------------------------------------------------------------------

def _safe_listdir(path: str) -> List[str]:
    """Return sorted listdir or empty list if path missing."""
    if not os.path.exists(path):
        print(f"[Warning] Path not found: {path}")
        return []
    return sorted(os.listdir(path))


def load_text(spec: Dict[str, Any]):
    """Load a list of captions or textual items.
    Supports JSON (a list of strings or list of dicts with 'caption' key) or
    plain txt files (one caption per line)."""
    path = spec["path"]
    fmt = spec.get("format", "json").lower()
    out: List[str] = []

    if fmt == "json":
        for fname in _safe_listdir(path):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    # list[str] or list[dict]
                    for item in data:
                        if isinstance(item, str):
                            out.append(item)
                        elif isinstance(item, dict):
                            out.append(item.get("caption", ""))
    else:  # plain text
        for fname in _safe_listdir(path):
            if fname.endswith(".txt"):
                with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                    out.extend([line.strip() for line in f if line.strip()])
    return out


def load_video(spec: Dict[str, Any]):
    """Return list of pre-saved video tensors (.pt files)."""
    base = spec["path"]
    pt_files = [os.path.join(base, f) for f in _safe_listdir(base) if f.endswith(".pt")]
    return pt_files



def load_audio(spec: Dict[str, Any]):
    base = spec["path"]
    wavs = [os.path.join(base, f) for f in _safe_listdir(base) if f.lower().endswith((".wav", ".flac", ".mp3"))]
    return wavs


def load_sensor(spec: Dict[str, Any]):
    base = spec["path"]
    files = [os.path.join(base, f) for f in _safe_listdir(base) if f.lower().endswith((".csv", ".json", ".npy"))]
    return files


# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------

def load_modalities(mod_cfg: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Iterate over all modalities defined in YAML and load them with the correct helper."""
    data: Dict[str, Any] = {}

    for name, spec in mod_cfg.items():
        mtype = spec.get("type", "").lower()
        try:
            if mtype in {"image", "video"}:
                data[name] = load_video(spec)
            elif mtype == "text":
                data[name] = load_text(spec)
            elif mtype == "audio":
                data[name] = load_audio(spec)
            elif mtype == "sensor":
                data[name] = load_sensor(spec)
            else:
                print(f"[Warning] Unknown modality type '{mtype}' for {name}; skipping.")
        except Exception as exc:
            print(f"[Error] Failed loading modality '{name}': {exc}")
            data[name] = []

    return data
