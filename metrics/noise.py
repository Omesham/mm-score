# metrics/noise.py
# ────────────────────────────────────────────────────────────
# NoiseMetric
#   • Generic out‑of‑range / outlier detection for *any* array‑like data
#   • Special cases for AUDIO and IMAGE (clipping / exposure checks)
#   • Detects “missing‑image” situations (e.g. more text than image samples)
#   • Produces an overall noise score ∈ [0,1]  + up‑to‑3 recommendations
# ────────────────────────────────────────────────────────────

from typing import Dict, Any, List                           # type hints
import numpy as np                                           # maths / arrays

# ────────────────────────────────────────────────────────────
# helper: map numeric score → qualitative label
# ────────────────────────────────────────────────────────────
def _grade(s: float) -> str:                                 # 0‑1 → string
    return ("EXCELLENT" if s >= .8 else                      # good ≥ 0.8
            "GOOD"      if s >= .6 else                      # …
            "FAIR"      if s >= .4 else
            "POOR"      if s >= .2 else
            "CRITICAL")

# thresholds used in per‑modality rules ---------------------
_CLIP_THR  = 0.99                                            # hard‑clip level
_SIL_STD   = 1e-4                                            # silence stdev
_DARK_P    = 0.05                                            # very dark pixel
_BRIGHT_P  = 0.95                                            # very bright pixel

# ────────────────────────────────────────────────────────────
class NoiseMetric:
    """Assesses data quality / noise for each detected modality."""

    def __init__(self, cfg: Dict[str, Any]):                 # cfg unused now
        self.cfg = cfg                                       # but kept for parity

    # ========================================================
    # main API called by the evaluator
    # ========================================================
    def evaluate(self, mods: Dict[str, Any]) -> Dict[str, Any]:
        per_mod_scores: List[float] = []                     # collect sub‑scores
        raw_recs:        List[str]  = []                     # collect rec strings

        # -------- 1) per‑modality quality checks ----------- #
        for name, data in mods.items():                      # iterate over modalities
            arr = self._first_array(data)                    # normalise container → np.array

            if name == "audio":                              # dedicated audio rule
                sc, rec = self._score_audio(arr)
            elif name == "image":                            # dedicated image rule
                sc, rec = self._score_image(arr)
            else:                                            # generic numeric data
                sc, rec = self._score_generic(arr)

            per_mod_scores.append(sc)                        # store modality score
            raw_recs.extend(rec)                             # collect its recs

        # -------- 2) dataset‑level missing‑image check ----- #
        if {"text", "image"} <= mods.keys():                 # we have both text & image modalities
            text_n  = self._count(mods["text"])              # how many text samples
            image_n = self._count(mods["image"])             # how many images
            if image_n < text_n * 0.8:                       # heuristic: >20 % missing
                raw_recs.append(f"{text_n - image_n} captions lack matching images; locate or regenerate images.")

        # -------- 3) aggregate results --------------------- #
        overall = float(np.mean(per_mod_scores)) if per_mod_scores else 1.0
        recs    = self._final_recs(overall, raw_recs)        # trim / default

        return {                                             # structure expected by framework
            "mm_score_summary": {
                "overall_score": overall,
                "overall_grade": _grade(overall),
                "component_scores": {"noise": overall}
            },
            "quality_assessment": {
                "recommendations": recs                      # ≤3 strings
            }
        }

    # ========================================================
    # per‑modality scoring helpers
    # ========================================================
    def _score_audio(self, wav: np.ndarray) -> tuple[float, List[str]]:
        wav = wav.flatten()                                  # 1‑D
        clip_ratio = np.mean(np.abs(wav) > _CLIP_THR)        # fraction clipped
        silence    = np.std(wav) < _SIL_STD                  # near‑zero variance
        score      = 1.0 - max(clip_ratio, 0)                # simple inverse

        recs: List[str] = []                                 # recommendations list
        if clip_ratio > 0.05:                                # >5 % clipped
            recs.append(f"~{clip_ratio*100:.1f}% of audio is clipped; lower gain.")
        if silence:
            recs.append("Audio track is nearly silent; check microphone or file corruption.")
        return max(0.0, score), recs

    def _score_image(self, img: np.ndarray) -> tuple[float, List[str]]:
        img_f = img.astype("float32") / 255.0                # 0‑1
        dark   = np.mean(img_f < _DARK_P)                    # % very dark pixels
        bright = np.mean(img_f > _BRIGHT_P)                  # % very bright pixels
        extreme= dark + bright                               # simplistic artefact proxy
        score  = 1.0 - extreme                               # lower extreme → better

        recs: List[str] = []
        if dark   > .25:                                     # >25 % very dark
            recs.append("Images are very dark; adjust lighting/exposure.")
        if bright > .25:                                     # >25 % very bright
            recs.append("Images are over‑exposed; reduce brightness.")
        return max(0.0, score), recs

    def _score_generic(self, arr: np.ndarray) -> tuple[float, List[str]]:
        flat     = arr.flatten()                             # 1‑D
        m, s     = np.mean(flat), np.std(flat) + 1e-8        # mean & std
        outliers = np.mean(np.abs(flat - m) > 2 * s)         # % >2σ
        score    = 1.0 - outliers                            # inverse
        recs     = []
        if outliers > .10:                                   # >10 % outliers
            recs.append(f"{outliers*100:.1f}% statistical outliers; investigate data quality for noise.")
        return max(0.0, score), recs

    # ========================================================
    # recommendation post‑processing
    # ========================================================
    @staticmethod
    def _final_recs(overall: float, raw: List[str]) -> List[str]:
        if raw:                                              # we already have specific recs
            return raw[:3]                                   # keep first‑3
        # otherwise add a generic line depending on quality
        if overall >= .8:
            return ["Noise levels are low—no fixes needed."]
        return ["Noise detected but no specific issues isolated."]

    # ========================================================
    # utility extractors
    # ========================================================
    @staticmethod
    def _first_array(obj: Any) -> np.ndarray:                # standardise data containers
        if isinstance(obj, dict) and "emb" in obj:           # pre‑computed embedding dict
            return obj["emb"]
        if isinstance(obj, list):                            # list → recurse on first
            return NoiseMetric._first_array(obj[0])
        if isinstance(obj, np.ndarray):                      # already ndarray
            return obj
        raise TypeError("Unsupported data container for noise metric.")

    @staticmethod
    def _count(obj: Any) -> int:                             # count samples in modality
        if isinstance(obj, dict) and "emb" in obj:           # embedding dict
            return obj["emb"].shape[0]
        if isinstance(obj, list):
            return len(obj)
        if isinstance(obj, np.ndarray):
            return obj.shape[0] if obj.ndim > 1 else 1
        return 1                                             # fallback
