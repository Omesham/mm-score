# mm_score_main.py
import yaml
import os
from metrics import get_metric
from loaders import load_modalities


class MMSCOREEvaluator:
    """Main runner that ties everything together: reads YAML, loads data, runs metrics."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            raw_config = f.read()

        # Replace ${PWD} with actual working directory
        resolved_config = raw_config.replace("${PWD}", os.getcwd())
        self.config = yaml.safe_load(resolved_config)

        # Load all modalities (text, video, audio, sensors, ...)
        self.modalities = load_modalities(self.config["modalities"])

        # Instantiate each metric listed in the YAML via the factory helper
        self.metrics = [get_metric(name)(self.config) for name in self.config["evaluation"]["metrics"]]

    def evaluate(self):
        results = {}
        for metric in self.metrics:
            results[metric.name()] = metric.evaluate(self.modalities)
        return results


if __name__ == "__main__":
    evaluator = MMSCOREEvaluator("config_mm_score.yaml")
    scorecard = evaluator.evaluate()
    print("MM‑SCORE results:\n", scorecard)
