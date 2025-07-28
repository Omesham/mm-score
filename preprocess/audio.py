from pathlib import Path
from typing import List, Iterable, Tuple

import torch
import torchaudio
from laion_clap import CLAP                         # pip install laion-clap
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText

from preprocess.base import BasePreprocessor
from Utils.loaders import UniversalDataLoader             # only used by run()

# ────────────────────────────────────────────────────────────
# fixed audio parameters
SR = 16_000
_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=1024, hop_length=512)

class AudioPreprocessor(BasePreprocessor):
    """
    Dual audio preprocessor:
    - CLAP: Embedding-based similarity (audio ↔ text)
    - SpeechT5: Text generation from audio (ASR)
    Outputs:
      - audio.npy → {"ids": [...], "emb": (N, 512)}
      - audio_transcripts.txt → ID + SpeechT5-generated text
    """

    def __init__(self, device: str = "cuda", use_clap: bool = True, use_speecht5: bool = True):
        self.device = device
        self.use_clap = use_clap
        self.use_speecht5 = use_speecht5

        if use_clap:
            self.clap_model = CLAP.get_model("music_audioset").to(device).eval()
        else:
            self.clap_model = None

        if use_speecht5:
            self.speecht5_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
            self.speecht5_model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr").to(device).eval()

        super().__init__(self.clap_model, device=device)

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
        if not self.use_clap:
            return torch.empty(0)
        if batch.dim() == 4:                            # (B,C,M,T)
            batch = batch.mean(1, keepdim=True)         # down‑mix channels
        return self.clap_model.encode_audio(batch.to(self.device))

    # ---------- ASR text generation -------------------------
    def generate_transcripts(self, paths: List[str], output_txt: Path):
        if not self.use_speecht5:
            return
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(output_txt, "w", encoding="utf-8") as f:
            for path in paths:
                wav, sr = torchaudio.load(path)
                if sr != SR:
                    wav = torchaudio.functional.resample(wav, sr, SR)
                input_values = self.speecht5_processor(wav.squeeze(0), sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
                with torch.no_grad():
                    predicted_ids = self.speecht5_model.generate(input_values)
                transcription = self.speecht5_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                f.write(f"{Path(path).stem}: {transcription}\n")

# ────────────────────────────────────────────────────────────
# Optional helper to run stand‑alone

def run(dataset_root: str, out_dir: str, device="cuda", use_clap=True, use_speecht5=True):
    detected = UniversalDataLoader.auto_detect_modalities(dataset_root)
    audio_files = detected.get("audio", [])
    if not audio_files:
        print("[AudioPreproc] No audio files found.")
        return

    pre = AudioPreprocessor(device=device, use_clap=use_clap, use_speecht5=use_speecht5)

    if use_clap:
        pre.encode_and_save(audio_files, Path(out_dir) / "audio.npy")
    if use_speecht5:
        pre.generate_transcripts(audio_files, Path(out_dir) / "audio_transcripts.txt")
