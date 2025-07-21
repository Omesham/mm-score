
# preprocess/base.py
import abc, pathlib, torch, numpy as np

class BasePreprocessor(abc.ABC):
    def __init__(self, model: torch.nn.Module, *, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

    @abc.abstractmethod
    def iter_samples(self, root: pathlib.Path):
        """Yield (sample_id, tensor) pairs for this modality."""

    @torch.inference_mode()
    def encode_and_save(self, root: pathlib.Path, out: pathlib.Path):
        out.mkdir(parents=True, exist_ok=True)
        for sid, x in self.iter_samples(root):
            emb = self.model(x.to(self.device)).cpu().numpy()
            np.save(out / f"{sid}.npy", emb)
