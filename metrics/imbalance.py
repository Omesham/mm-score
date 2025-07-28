# metrics/imbalance_entropy.py
from typing import Dict, Any
import numpy as np
from scipy.stats import entropy

class EntropyImbalanceMetric:
    """
    Entropy-based modality imbalance:
    - Computes the entropy of modality distribution across all samples.
    - Higher entropy → more balanced.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def evaluate(self, mods: Dict[str, Any]) -> Dict[str, Any]:
        # Count number of samples per modality
        counts = {name: self._count(data) for name, data in mods.items()}
        total = sum(counts.values())

        if total == 0 or len(counts) < 2:
            score = 1.0
            grade = "EXCELLENT"
            recs = ["Not enough data to assess entropy-based imbalance."]
        else:
            probs = np.array([v / total for v in counts.values()])
            ent = entropy(probs, base=len(probs))  # normalized entropy [0,1]
            score = ent
            grade = self._grade(score)
            recs = self._make_recs(score)

        return {
            "mm_score_summary": {
                "overall_score": score,
                "overall_grade": grade,
                "component_scores": {"entropy_imbalance": score}
            },
            "quality_assessment": {
                "recommendations": recs
            }
        }

    @staticmethod
    def _count(obj: Any) -> int:
        if isinstance(obj, dict) and "emb" in obj:
            return obj["emb"].shape[0]
        if isinstance(obj, list):
            return len(obj)
        if isinstance(obj, np.ndarray):
            return obj.shape[0]
        return 1

    @staticmethod
    def _grade(s: float) -> str:
        return ("EXCELLENT" if s >= 0.8 else
                "GOOD"      if s >= 0.6 else
                "FAIR"      if s >= 0.4 else
                "POOR"      if s >= 0.2 else
                "CRITICAL")

    @staticmethod
    def _make_recs(score: float):
        if score < 0.2:
            return ["Severe imbalance: Consider upsampling underrepresented modalities."]
        if score < 0.4:
            return ["Noticeable imbalance: Try rebalancing modality presence."]
        if score < 0.6:
            return ["Mild imbalance: Balance could be improved."]
        return ["Modality distribution is well balanced."]
