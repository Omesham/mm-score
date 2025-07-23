# preprocess/audio.py
from pathlib import Path
from typing import List, Iterable, Tuple
import torch, torchaudio
from clap_module import CLAP # pip install laion-clap

from preprocess.base import BasePreprocessor

_tf = torchaudio.transforms.MelSpectrogram(            # quick spectrogram
        sample_rate=16_000, n_fft=1024, hop_length=512)

class AudioPreprocessor(BasePreprocessor):
    def __init__(self, device="cuda"):
        model = CLAP(pretrained="openai").to(device).eval()
        super().__init__(model, device=device)

    def iter_samples(self, paths: List[str]) -> Iterable[Tuple[str, torch.Tensor]]:
        for p in paths:
            wav, sr = torchaudio.load(p)
            if sr != 16_000:
                wav = torchaudio.functional.resample(wav, sr, 16_000)
            mel = _tf(wav)            # (channel, mel, time)
            yield Path(p).stem, mel.unsqueeze(0)  # add batch‑dim
