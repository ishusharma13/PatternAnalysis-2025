import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyBackbone(nn.Module):
    """
    Temporary lightweight backbone that mimics MambaIRv2 features.
    Replace this later with the real MambaIRv2 once installed.
    """
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class ColorizationHead(nn.Module):
    def __init__(self, in_channels, mid=128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 3, kernel_size=1)
        )

    def forward(self, x):
        return torch.sigmoid(self.decoder(x))

class MambaColorizer(nn.Module):
    """
    Wraps a (temporary) backbone + decoder head.
    Once MambaIRv2 is available, swap DummyBackbone with the pretrained one.
    """
    def __init__(self, freeze_backbone=False):
        super().__init__()
        self.backbone = DummyBackbone()
        self.head = ColorizationHead(in_channels=64)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, gray):
        # Expand grayscale 1→3 channels for backbone
        x3 = gray.repeat(1,3,1,1)
        feat = self.backbone(x3)
        color = self.head(feat)
        # Resize to match input
        if color.shape[2:] != gray.shape[2:]:
            color = F.interpolate(color, size=gray.shape[2:], mode='bilinear', align_corners=False)
        return color
