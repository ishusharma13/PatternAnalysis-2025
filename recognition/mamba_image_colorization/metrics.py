import torch
import torch.nn.functional as F
from math import log10
from skimage.metrics import structural_similarity as ssim_metric

def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100
    return 10 * log10(1 / mse.item())

def ssim(pred, target):
    pred_np = pred.squeeze(0).permute(1,2,0).cpu().numpy()
    target_np = target.squeeze(0).permute(1,2,0).cpu().numpy()
    return ssim_metric(target_np, pred_np, channel_axis=-1)
