# train.py (updated for Rangpur / GPU-ready)
import argparse
import os
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloader import GrayscaleColorDataset
from modules import MambaColorizer
from metrics import psnr, ssim
import lpips
from torchvision.transforms import Compose, ToTensor

# --------------------------
# Utility functions
# --------------------------
def save_val_image(tensor, fname, out_dir="./val_results"):
    os.makedirs(out_dir, exist_ok=True)
    from torchvision.transforms.functional import to_pil_image
    img = to_pil_image(tensor.squeeze(0).clamp(0, 1))
    img.save(os.path.join(out_dir, fname))

def log_metrics_to_csv(epoch, train_loss, psnr_score, ssim_score, lpips_score, filepath="./logs/metrics.csv"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "train_loss", "psnr", "ssim", "lpips"])
        writer.writerow([epoch, train_loss, psnr_score, ssim_score, lpips_score])

# --------------------------
# Training / Validation
# --------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, p_loss_fn, device):
    model.train()
    total_loss = 0
    for gray, color in tqdm(loader, desc="Training"):
        gray, color = gray.to(device), color.to(device)
        pred = model(gray)
        rec_loss = loss_fn(pred, color)
        perceptual_loss = p_loss_fn(pred, color).mean()
        loss = rec_loss + 0.1 * perceptual_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, loss_fn, device):
    model.eval()
    psnr_list, ssim_list, lpips_list = [], [], []

    with torch.no_grad():
        for gray, color in tqdm(loader, desc="Validating"):
            gray, color = gray.to(device), color.to(device)
            pred = model(gray)
            psnr_list.append(psnr(pred, color))
            ssim_list.append(ssim(pred, color))
            lpips_list.append(loss_fn(pred, color).mean().item())
    avg_psnr = sum(psnr_list)/len(psnr_list) if psnr_list else 0
    avg_ssim = sum(ssim_list)/len(ssim_list) if ssim_list else 0
    avg_lpips = sum(lpips_list)/len(lpips_list) if lpips_list else 0
    return avg_psnr, avg_ssim, avg_lpips

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./", help="dataset root folder")
    parser.add_argument("--bs", type=int, default=4, help="batch size")
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs")
    parser.add_argument("--size", type=int, default=128, help="resize images")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--vendor", type=str, default="vim", choices=["vim","mambairv2","dummy"])
    parser.add_argument("--pretrained", action="store_true", help="use pretrained backbone")
    parser.add_argument("--local_rank", type=int, default=-1, help="for DDP")
    args = parser.parse_args()

    # Device
    distributed = args.local_rank != -1
    if distributed:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "Distributed:", distributed)

    # Transforms
    transform = Compose([ToTensor()])

    # Datasets
    train_ds = GrayscaleColorDataset(
        gray_folder=os.path.join(args.data, "coco_train_final/gray"),
        color_folder=os.path.join(args.data, "coco_train_final/color"),
        transform=transform
    )
    val_ds = GrayscaleColorDataset(
        gray_folder=os.path.join(args.data, "coco_val_final/gray"),
        color_folder=os.path.join(args.data, "coco_val_final/color"),
        transform=transform
    )

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = MambaColorizer(vendor=args.vendor, pretrained=args.pretrained).to(device)
    if distributed:
        model = DDP(model, device_ids=[args.local_rank])

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-6)
    loss_fn = nn.L1Loss()
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)

    os.makedirs("./checkpoints", exist_ok=True)
    best_psnr = 0.0

    # Training loop
    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, perceptual_loss_fn, device)
        avg_psnr, avg_ssim, avg_lpips = validate(model, val_loader, perceptual_loss_fn, device)
        print(f"\nEpoch [{epoch+1}/{args.epochs}] TrainLoss: {avg_loss:.4f} PSNR: {avg_psnr:.2f} SSIM: {avg_ssim:.4f} LPIPS: {avg_lpips:.4f}")

        log_metrics_to_csv(epoch+1, avg_loss, avg_psnr, avg_ssim, avg_lpips)

        # Save checkpoint
        ckpt_path = os.path.join("./checkpoints", f"ckpt_epoch{epoch+1}.pth")
        torch.save({
            "epoch": epoch+1,
            "model_state": model.module.state_dict() if distributed else model.state_dict(),
            "optim_state": optimizer.state_dict()
        }, ckpt_path)
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                "epoch": epoch+1,
                "model_state": model.module.state_dict() if distributed else model.state_dict(),
                "optim_state": optimizer.state_dict()
            }, "./checkpoints/best.pth")

if __name__ == "__main__":
    main()
