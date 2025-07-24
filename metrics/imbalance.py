# metrics/imbalance.py
# ────────────────────────────────────────────────────────────
# ImbalanceMetric
#   • Looks at how many samples each detected modality has.
#   • Penalises severe under‑representation.
#   • Gives up‑to‑3 concrete, human‑readable suggestions.
# ────────────────────────────────────────────────────────────
from typing import Dict, Any, List
import numpy as np

# ---------- helpers ------------------------------------------------------
def _grade(s: float) -> str:
    return ("EXCELLENT" if s >= .8 else
            "GOOD"      if s >= .6 else
            "FAIR"      if s >= .4 else
            "POOR"      if s >= .2 else
            "CRITICAL")

def _count(obj: Any) -> int:
    """Return #samples in whatever container the loader gave us."""
    if isinstance(obj, dict) and "emb" in obj:     # pre‑computed embeddings
        return obj["emb"].shape[0]
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, np.ndarray):
        return obj.shape[0] if obj.ndim > 1 else 1
    return 1                                       # fallback for scalars etc.

# ────────────────────────────────────────────────────────────
class ImbalanceMetric:
    """Quantity‑imbalance across modalities (simple but useful)."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg       # (not used for now – placeholder)

    # ========================================================
    # public API
    # ========================================================
    def evaluate(self, mods: Dict[str, Any]) -> Dict[str, Any]:
        # -------- collect counts for every modality ----------
        counts = {name: _count(data) for name, data in mods.items()}

        if len(counts) < 2:                     # single‑modality dataset
            score = 1.0                         # trivially balanced
            recs  = ["Single modality – imbalance not applicable."]
        else:
            mx, mn = max(counts.values()), min(counts.values())
            ratio  = mx / (mn + 1e-8)           # avoid div‑by‑zero

            # simple inverse‑ratio mapping  (ratio 1 → score 1, ratio ≥10 → score →0)
            score = max(0.0, min(1.0, 1.0 / ratio))

            # actionable feedback ---------------------------------------
            recs = self._make_recs(counts, ratio)[:3]  # keep first 3

        return {
            "mm_score_summary": {
                "overall_score": score,
                "overall_grade": _grade(score),
                "component_scores": {"imbalance": score}
            },
            "quality_assessment": {
                "recommendations": recs
            }
        }

    # ========================================================
    # recommendation logic
    # ========================================================
    @staticmethod
    def _make_recs(counts: Dict[str, int], ratio: float) -> List[str]:
        recs: List[str] = []

        # Find worst‑represented modality (fewest samples)
        worst_mod = min(counts, key=counts.get)
        best_mod  = max(counts, key=counts.get)

        # Heuristics: suggest bringing low modality up to ~80 % of max
        target = int(counts[best_mod] * 0.8)
        deficit = max(0, target - counts[worst_mod])

        if ratio > 10:
            recs.append(f"{worst_mod} has <10 % the samples of {best_mod}; "
                        f"collect roughly +{deficit} more {worst_mod} examples.")
        elif ratio > 5:
            recs.append(f"Noticeable imbalance: {worst_mod}:{best_mod} ≈ 1:{ratio:.1f}. "
                        f"Consider boosting {worst_mod} by ~{deficit}.")
        elif ratio > 2:
            recs.append(f"Mild imbalance detected (ratio ≈ {ratio:.1f}). "
                        "Top‑up the smaller modalities to improve fairness.")

        if len(counts) >= 3 and ratio > 5:
            recs.append("Consider weighted loss functions or over‑sampling during training.")

        if not recs:
            recs.append("Sample counts are well balanced – no action needed.")

        return recs
