# ─────────────────────────────  preprocess/text.py  ───────────────────────────
"""
TextPreprocessor
----------------
Encodes **every individual caption / sentence / cell** found in the supplied text
files with CLIP (ViT-B/32 tokenizer).

Outputs:
    text.npy  →  {"ids": [...], "emb": ndarray (N, 512)}
"""

from pathlib import Path
from typing import List, Iterable, Tuple
import json, csv
import torch
import clip                                   # only for its tokenizer
from preprocess.clip_base import CLIPPreprocessor
from typing import Optional

class TextPreprocessor(CLIPPreprocessor):
    """One sample per caption / line / cell."""

    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_strings(obj):
        """Recursively pull every string out of nested JSON."""
        if isinstance(obj, str):
            return [obj]
        if isinstance(obj, dict):
            out = []
            for v in obj.values():
                out.extend(TextPreprocessor._extract_strings(v))
            return out
        if isinstance(obj, (list, tuple, set)):
            out = []
            for v in obj:
                out.extend(TextPreprocessor._extract_strings(v))
            return out
        return []

    # ------------------------------------------------------------------ #
    def iter_samples(self, paths: List[str], max_items: Optional[int] = None) -> Iterable[Tuple[str, torch.Tensor]]:        
        """
        Yield (uid, token_ids) for **every caption/sentence** inside each file.
        """
        count = 0  # ← Initialize counter
        for p in paths:
            print("Printing text path:", p)
            p = Path(p)
            stem = p.stem.lower()
            ext = p.suffix.lower()

            # ---------- JSON (COCO or generic) -------------------------
            if ext == ".json":
                with p.open("r", encoding="utf-8") as f:
                    content = json.load(f)

                # COCO-annotations branch
                if isinstance(content, dict) and "annotations" in content:
                    for idx, ann in enumerate(content["annotations"]):
                        if max_items is not None and count >= max_items:
                            return
                        caption = ann.get("caption") or ann.get("text")
                        if caption:
                            yield f"{stem}_{idx}", clip.tokenize(caption, truncate=True)
                            count += 1

                # List branch
                elif isinstance(content, list):
                    for idx, item in enumerate(content):
                        if max_items and count >= max_items:
                            return
                        if isinstance(item, str) and item.strip():
                            yield f"{stem}_{idx}", clip.tokenize(item, truncate=True)
                            count += 1
                        elif isinstance(item, dict):
                            for k, v in item.items():
                                if isinstance(v, str) and v.strip():
                                    yield f"{stem}_{idx}_{k}", clip.tokenize(v, truncate=True)
                                    count += 1

                # Fallback – extract every string
                for idx, s in enumerate(self._extract_strings(content)):
                    if max_items and count >= max_items:
                        return
                    yield f"{stem}_{idx}", clip.tokenize(s, truncate=True)
                    count += 1
            # ---------- JSON-Lines -------------------------------------
            elif ext == ".jsonl":
                with p.open("r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        for s in self._extract_strings(obj):
                            yield f"{stem}_{idx}", clip.tokenize(s, truncate=True)

            # ---------- Plain TXT --------------------------------------
            elif ext == ".txt":
                with p.open("r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if line:
                            yield f"{stem}_{idx}", clip.tokenize(line, truncate=True)

            # ---------- CSV / TSV --------------------------------------
            elif ext in {".csv", ".tsv"}:
                sep = "\t" if ext == ".tsv" else ","
                with p.open("r", encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter=sep)
                    for r_idx, row in enumerate(reader):
                        for c_idx, cell in enumerate(row):
                            cell = cell.strip()
                            if cell:
                                yield f"{stem}_{r_idx}_{c_idx}", clip.tokenize(cell, truncate=True)

            # ---------- Anything else ----------------------------------
            else:
                text = p.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    yield stem, clip.tokenize(text, truncate=True)


# ---------------------------- quick test ------------------------------------
def run(dataset_root: str, out_dir: str, device: str = "cuda"):
    """
    python -m preprocess.text run /data/COCO /data/embeddings
    """
    from Utils.loaders import UniversalDataLoader

    text_files = UniversalDataLoader.auto_detect_modalities(dataset_root).get("text", [])
    if not text_files:
        print("[TextPreproc] No text files found.")
        return

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    TextPreprocessor(device=device).encode_and_save(text_files, out_dir / "text.npy")
# ─────────────────────────────────────────────────────────────────────────────
