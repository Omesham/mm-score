from abc import ABC, abstractmethod

class Metric(ABC):
    """
    Abstract base class for all MM-SCORE metrics.
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def name(self):
        """
        Return the name of the metric (e.g., 'alignment', 'noise').
        """
        pass

    @abstractmethod
    def evaluate(self, modality_dict):
        """
        Compute the metric using the loaded modality data.
        Args:
            modality_dict (dict): {modality_name: loaded_data}
        Returns:
            dict: metric result, e.g. {"video_text_similarity": 0.92}
        """
        pass
