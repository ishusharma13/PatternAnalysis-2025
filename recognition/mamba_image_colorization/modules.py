import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing real MambaIR backbone
try:
    from model_zoo.mambairv2 import buildMambaIRv2Small
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

class ColorizationHead(nn.Module):
    def __init__(self, in_channels=64, mid=128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 3, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.decoder(x))

class MambaColorizer(nn.Module):
    def __init__(self, freeze_backbone=False):
        super().__init__()
        
        if MAMBA_AVAILABLE:
            print("✅ Using MambaIRv2 backbone")
            self.backbone = buildMambaIRv2Small(pretrained=False)
            out_channels = 64  # backbone output feature size
        else:
            print("⚠️ MambaIR not installed — using dummy CNN")
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU()
            )
            out_channels = 64

        self.head = ColorizationHead(in_channels=out_channels)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, gray):
        x3 = gray.repeat(1,3,1,1)
        feat = self.backbone(x3)
        color = self.head(feat)
        if color.shape[2:] != gray.shape[2:]:
            color = F.interpolate(color, size=gray.shape[2:], mode='bilinear', align_corners=False)
        return color
