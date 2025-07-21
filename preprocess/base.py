
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
    def encode_and_save(self, paths: List[str], out_file: Path):
        """
        Encode every sample in `paths` and store **one** .npy file:
            {"ids": [id₁, id₂, …], "emb": (N, D) array}
        """
        feats, ids = [], []
        for sid, x in self.iter_samples(paths):
            vec = self.model(x.to(self.device)).cpu().numpy()  # (1, D)
            feats.append(vec.squeeze())                        # → (D,)
            ids.append(sid)

        emb_matrix = np.stack(feats)                           # (N, D)
        np.save(out_file, {"ids": ids, "emb": emb_matrix})

