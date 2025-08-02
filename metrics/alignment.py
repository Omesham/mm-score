# metrics/alignment.py
# ────────────────────────────────────────────────────────────
# AlignmentMetric
#   • Semantic (image ↔ text) with CLIP
#   • Temporal (video / audio / sensor / …) by cross‑correlation
#   • Produces an overall alignment score ∈ [0,1] **and**
#     a quality‑assessment section listing up‑to‑3 fixes.
# ────────────────────────────────────────────────────────────
from typing import Dict, Any, List
import numpy as np
from typing import Union

# ---------- optional CLIP import ----------------------------------------
try:
    import torch, clip
    from PIL import Image
    _CLIP_OK = True
except ImportError:
    _CLIP_OK = False

# ---------- helpers ------------------------------------------------------
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def _grade(s: float) -> str:
    return ("EXCELLENT" if s >= .8 else
            "GOOD"      if s >= .6 else
            "FAIR"      if s >= .4 else
            "POOR"      if s >= .2 else
            "CRITICAL")

# ────────────────────────────────────────────────────────────
class AlignmentMetric:
    TEMPORAL = {"video", "audio", "sensor", "temperature",
                "heart", "brain", "eyes", "emotions"}

    # ------------------------------------------------------------------
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.clip_model, self.clip_pre = (None, None)
        if _CLIP_OK:
            self.clip_model, self.clip_pre = clip.load("ViT-B/32",
                                                       device="cpu", jit=False)
            self.clip_model.eval()
        self._temporal_length_range = None
        self._semantic_length_mismatch = False

    # ------------------------------------------------------------------
    def evaluate(self, mods: Dict[str, Any]) -> Dict[str, Any]:
        sem = self._semantic(mods)          # None or [0,1]
        tmp = self._temporal(mods)          # None or [0,1]

        if sem is not None and tmp is not None:
            align = 0.5 * sem + 0.5 * tmp
        elif sem is not None:
            align = sem
        elif tmp is not None:
            align = tmp
        else:
            align = 0.0

        recs = self._make_recs(sem, tmp, mods)[:3]   # top‑3

        return {
            "mm_score_summary": {
                "overall_score": align,
                "overall_grade": _grade(align),
                "component_scores": {"alignment": align}
            },
            "quality_assessment": {
                "recommendations": recs
            }
        }

  
    # ─────────────────────────────────────────────────────────
    # semantic alignment (image ↔ text)
    # ─────────────────────────────────────────────────────────
    def _semantic(self, mods: Dict[str, Any]) -> Union[float, None]:
        # Need both modalities
        if not {"image", "text"} <= mods.keys():
            return None

        images = mods["image"]
        texts  = mods["text"]

        # ---------- 1) Direct use of pre-computed CLIP embeddings ----------
        # Dict format: {"ids": [...], "emb": ndarray (N,512)}
        if isinstance(images, dict) and "emb" in images \
           and isinstance(texts,  dict) and "emb" in texts:
            v_img, v_txt = images["emb"][0], texts["emb"][0]
            return (_cos(v_img, v_txt) + 1) / 2       # → [0,1]

        # Plain ndarray format: (N,512)
        if isinstance(images, np.ndarray) and images.ndim == 2 \
           and isinstance(texts,  np.ndarray) and texts.ndim == 2 \
           and images.shape[1] == texts.shape[1] == 512:
            return (_cos(images[0], texts[0]) + 1) / 2

        # ---------- 2) Fallback: raw image + raw caption -------------------
        if not _CLIP_OK:          # CLIP not available → cannot compute
            return None

        # If lists are mismatched in length, flag recommendation
        if isinstance(images, list) and isinstance(texts, list) and len(images) != len(texts):
            self._semantic_length_mismatch = True

        img = self._first_array(images)   # (H,W,3) uint8
        txt = self._first_text(texts)     # str

        with torch.no_grad():
            img_t = self.clip_pre(Image.fromarray(img)).unsqueeze(0)
            txt_t = clip.tokenize([txt])
            v_img = self.clip_model.encode_image(img_t).cpu().numpy()[0]
            v_txt = self.clip_model.encode_text(txt_t).cpu().numpy()[0]
        return (_cos(v_img, v_txt) + 1) / 2


    # ─────────────────────────────────────────────────────────
    # temporal alignment  (cross‑corr on mean signal)
    # ─────────────────────────────────────────────────────────
    def _temporal(self, mods: Dict[str, Any]) -> Union[float, None]:
        traces = [self._first_array(v) for k, v in mods.items()
                  if k in self.TEMPORAL]

        if len(traces) < 2:
            return None

        lengths = [t.shape[0] for t in traces]
        min_len, max_len = min(lengths), max(lengths)
        self._temporal_length_range = (min_len, max_len)

        L = min_len  # truncate to match
        traces = [(t[:L].mean(1) if t.ndim > 1 else t[:L]) for t in traces]

        sims = [abs(np.corrcoef(traces[i], traces[j])[0, 1])
                for i in range(len(traces))
                for j in range(i+1, len(traces))]
        return float(np.mean(sims))

    # ─────────────────────────────────────────────────────────
    # actionable feedback
    # ─────────────────────────────────────────────────────────
    # metrics/alignment.py  – inside AlignmentMetric
    # -------------------------------------------------------------
    def _make_recs(self, sem, tmp, mods) -> List[str]:
        """
        Build a short list of actionable suggestions based on which
        components were *really* computed.
    
           sem  → semantic score  (None if image+text missing or CLIP disabled)
           tmp  → temporal score  (None if <2 temporal modalities)
           mods → dict of loaded modalities
        """
        recs: List[str] = []
    
        # ---------- semantic alignment recommendations --------------------
        if sem is not None and sem < 0.6 and {"image", "text"} <= mods.keys():
            recs.append("Image captions and texts look weakly aligned "
                        f"(CLIP-score ≈ {sem:.2f}). Review descriptions.")
    
        if self._semantic_length_mismatch:
            recs.append("Image and text modalities have mismatched lengths. "
                        "Ensure exactly one caption per image—or consolidate multiple captions.")
    
        # ---------- temporal alignment recommendations -------------------
        if tmp is not None:
            if tmp < 0.6:
                recs.append("Temporal streams appear out-of-sync "
                            f"(corr ≈ {tmp:.2f}). Check timestamps or trimming.")
    
            # only if _temporal_length_range exists *and* is a tuple
            if getattr(self, "_temporal_length_range", None):
                min_len, max_len = self._temporal_length_range
                if max_len - min_len > 10:
                    recs.append("Temporal streams have inconsistent lengths "
                                f"(range: {min_len}–{max_len}). "
                                "Consider trimming or resampling to align durations.")
    
        # ---------- audio-specific check ---------------------------------
        if "audio" in mods and self._is_silent(mods["audio"]):
            recs.append("Many audio segments are near-silent; verify recordings.")
    
        # ---------- default ------------------------------------------------
        if not recs:
            recs.append("Alignment is strong—no immediate fixes needed.")
    
        return recs


    # quick silence check for audio arrays --------------------
    @staticmethod
    def _is_silent(a) -> bool:
        arr = AlignmentMetric._first_array(a)
        return arr.std() < 1e-3

    # ─────────────────────────────────────────────────────────
    # utility extractors
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _first_array(obj):
        if isinstance(obj, dict) and "emb" in obj:
            return obj["emb"]
        if isinstance(obj, list):
            return AlignmentMetric._first_array(obj[0])
        if isinstance(obj, np.ndarray):
            return obj
        raise TypeError("Unsupported array type")

    @staticmethod
    def _first_text(obj):
        if isinstance(obj, list) and obj:
            return str(obj[0])
        if isinstance(obj, dict) and obj.get("annotations"):
            return str(obj["annotations"][0])
        if isinstance(obj, str):
            return obj
        return "sample text"
