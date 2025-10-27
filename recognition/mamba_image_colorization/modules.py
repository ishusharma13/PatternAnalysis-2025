# modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model_zoo.mambairv2 import buildMambaIRv2Small
    MAMBA_AVAILABLE = True
except Exception:
    print("⚠️ MambaIRv2 not found — fallback to dummy CNN.")
    MAMBA_AVAILABLE = False

class ColorizationHead(nn.Module):
    def __init__(self, in_ch=64, mid=128):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 3, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.dec(x))

class MambaColorizer(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super().__init__()

        if MAMBA_AVAILABLE:
            self.backbone = buildMambaIRv2Small(pretrained=pretrained)
            out_ch = getattr(self.backbone, 'out_channels', 64)
            print("✅ Using MambaIRv2 backbone")
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU()
            )
            out_ch = 64
            print("⚠️ Using dummy backbone")

        self.head = ColorizationHead(in_ch=out_ch)

        if freeze_backbone and MAMBA_AVAILABLE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("✅ Backbone frozen — Fine-tuning only color head")

    def forward(self, gray):
        x = gray.repeat(1,3,1,1)
        feat = self.backbone(x)

        if isinstance(feat, dict):
            feat = list(feat.values())[-1]

        out = self.head(feat)
        out = F.interpolate(out, size=gray.shape[2:], mode='bilinear', align_corners=False)
        return out
