# models/timesformer_encoder.py
import torch
import torch.nn as nn
from timesformer.models.vit import TimeSformer  # from official FB TimeSformer repo

class TimeSformerEncoder(nn.Module):
    def __init__(self, img_size=224, num_frames=8, pretrained_model='diving48', device='cuda'):
        super().__init__()
        self.model = TimeSformer(
            img_size=img_size,
            num_classes=400,         # dummy, we use encoder only
            num_frames=num_frames,
            attention_type="divided_space_time"
        )
        ckpt = torch.hub.load_state_dict_from_url(
            f"https://dl.fbaipublicfiles.com/timesformer/models/{pretrained_model}.pth",
            map_location=device
        )
        self.model.load_state_dict(ckpt)
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.to(self.device)
        with torch.no_grad():
            embeddings = self.model(x)  # (B, num_classes) but we extract encoder output
        return embeddings


