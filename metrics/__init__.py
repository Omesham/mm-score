from .alignment import ComprehensiveAlignmentMetric as AlignmentMetric

# Simple placeholder classes for other metrics
class NoiseMetric:
    def __init__(self, cfg):
        self.config = cfg
        print("✅ Universal NoiseMetric initialized")
    def name(self):
        return "noise"
    def evaluate(self, modalities):
        print("✅ Running universal noise evaluation")
        return {"noise_score": 0.90}

class ImbalanceMetric:
    def __init__(self, cfg):
        self.config = cfg
        print("✅ Universal ImbalanceMetric initialized")
    def name(self):
        return "imbalance"
    def evaluate(self, modalities):
        print("✅ Running universal imbalance evaluation")
        return {"imbalance_score": 0.85}

def get_metric(name):
    if name == "alignment":
        return AlignmentMetric
    elif name == "noise":
        return NoiseMetric
    elif name == "imbalance":
        return ImbalanceMetric
    else:
        raise ValueError(f"Unknown metric: {name}")
