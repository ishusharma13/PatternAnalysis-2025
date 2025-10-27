# modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try Vim (vision-mamba) first, then MambaIRv2, else fallback to dummy
VENDOR = None
try:
    # vision-mamba expected import locations (weights available)
    from vim.models import build_vim  # if installed as package
    VENDOR = 'vim'
except Exception:
    try:
        # try local repo path used by some forks
        from vismamba.model_zoo import build_vim
        VENDOR = 'vim'
    except Exception:
        try:
            from model_zoo.mambairv2 import buildMambaIRv2Small as build_mambair
            VENDOR = 'mambairv2'
        except Exception:
            VENDOR = 'dummy'

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
    def __init__(self, vendor='auto', pretrained=False, freeze_backbone=False):
        """
        vendor: 'vim' | 'mambairv2' | 'dummy' | 'auto'
        """
        super().__init__()
        self.vendor = vendor if vendor != 'auto' else VENDOR

        if self.vendor == 'vim':
            try:
                # user must pip install vision-mamba or clone and pip install -e
                # build_vim should return a model that accepts Bx3xHxW and returns features
                backbone = build_vim(pretrained=pretrained)
                self.backbone = backbone
                out_ch = getattr(backbone, 'out_channels', 64)
            except Exception as e:
                print("⚠️ Vim import failed:", e)
                self._use_dummy()
        elif self.vendor == 'mambairv2':
            try:
                from model_zoo.mambairv2 import buildMambaIRv2Small
                self.backbone = buildMambaIRv2Small(pretrained=pretrained)
                out_ch = getattr(self.backbone, 'out_channels', 64)
            except Exception as e:
                print("⚠️ MambaIR import failed:", e)
                self._use_dummy()
        else:
            self._use_dummy()

        self.head = ColorizationHead(in_channels=out_ch)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _use_dummy(self):
        # fallback simple encoder that returns Bx64xHxW
        self.vendor = 'dummy'
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.out_ch = 64

    def forward(self, gray):
        # gray: Bx1xHxW -> to 3-channel
        x = gray.repeat(1,3,1,1)
        feat = self.backbone(x)
        # if backbone returns dict, pick a tensor
        if isinstance(feat, dict):
            tensors = [v for v in feat.values() if isinstance(v, torch.Tensor)]
            feat = tensors[-1] if tensors else list(feat.values())[0]
        # adapt channels if needed
        if feat.shape[1] != self.head.decoder[0].in_channels:
            # lazy adapter
            if not hasattr(self, '_adapter'):
                self._adapter = nn.Conv2d(feat.shape[1], self.head.decoder[0].in_channels, 1).to(feat.device)
            feat = self._adapter(feat)
        out = self.head(feat)
        if out.shape[2:] != gray.shape[2:]:
            out = F.interpolate(out, size=gray.shape[2:], mode='bilinear', align_corners=False)
        return out
