# preprocess/audio.py
from pathlib import Path
from typing import List, Iterable, Tuple

import torch
import torchaudio
from laion_clap import CLAP                         # pip install laion-clap

from preprocess.base import BasePreprocessor
from loaders import UniversalDataLoader             # only used by run()

# ────────────────────────────────────────────────────────────
# fixed audio parameters
SR = 16_000
_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=1024, hop_length=512)

class AudioPreprocessor(BasePreprocessor):
    """
    Encode *.wav / *.flac / … files with LAION–CLAP → one audio.npy
        {"ids": [...], "emb": (N, 512)}
    """

    def __init__(self, device: str = "cuda"):
        model = CLAP.get_model("music_audioset").to(device).eval()
        super().__init__(model, device=device)

    # ---------- Loader hook ----------------------------------
    def iter_samples(self, paths: List[str]) -> Iterable[Tuple[str, torch.Tensor]]:
        for p in paths:
            wav, sr = torchaudio.load(p)
            if sr != SR:
                wav = torchaudio.functional.resample(wav, sr, SR)
            mel = _mel(wav)                             # (chan, n_mels, time)
            yield Path(p).stem, mel.unsqueeze(0)        # + batch dim

    # ---------- vector extraction override ------------------
    def get_vector(self, batch: torch.Tensor) -> torch.Tensor:
        # CLAP expects mono log‑mel (B, 1, n_mels, T)
        if batch.dim() == 4:                            # (B,C,M,T)
            batch = batch.mean(1, keepdim=True)         # down‑mix channels
        return self.model.encode_audio(batch.to(self.device))


# ────────────────────────────────────────────────────────────
# Optional helper to run stand‑alone (image.py & text.py use the same idea)
def run(dataset_root: str, out_dir: str, device="cuda"):
    detected = UniversalDataLoader.auto_detect_modalities(dataset_root)
    audio_files = detected.get("audio", [])
    if not audio_files:
        print("[AudioPreproc] No audio files found.")
        return

    pre = AudioPreprocessor(device=device)
    pre.encode_and_save(audio_files, Path(out_dir) / "audio.npy")
