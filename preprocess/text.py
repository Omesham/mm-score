# preprocess/text.py
from pathlib import Path
from typing import List, Iterable, Tuple
import torch
import clip                               # we only need its tokenizer
from preprocess.clip_base import CLIPPreprocessor


class TextPreprocessor(CLIPPreprocessor):
    """
    Encode every .txt / .json / .csv cell etc. with CLIP ViT‑B/32.
    Output single text.npy :  {"ids": [...], "emb": (N, 512)}
    """

    def iter_samples(self, paths: List[str]) -> Iterable[Tuple[str, torch.Tensor]]:
        """
        Yield (file_stem, token_ids) for each text file in `paths`.
        CLIP's tokenizer returns shape (1, 77) long – batch dim already present.
        """
        for p in paths:
            text = Path(p).read_text(encoding="utf‑8", errors="ignore").strip()
            if not text:
                continue                    # skip empty files
            tokens = clip.tokenize(text, truncate=True)   # (1, 77) long
            yield Path(p).stem, tokens


# ----------------------------------------------------------------------
# Optional helper to run stand‑alone (same pattern as image.py) for quick sanity check/testing
def run(dataset_root: str, out_dir: str, device="cuda"):
    from loaders import UniversalDataLoader

    detected = UniversalDataLoader.auto_detect_modalities(dataset_root)
    text_files = detected.get("text", [])
    if not text_files:
        print("[TextPreproc] No text files found.")
        return

    pre = TextPreprocessor(device=device)
    pre.encode_and_save(text_files, Path(out_dir) / "text.npy")

