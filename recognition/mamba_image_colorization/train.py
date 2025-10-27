# train.py (final)
import argparse
import os
import csv
from metrics import psnr, ssim
import lpips
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from dataset import PairedGrayColorDataset
from modules import MambaColorizer
from torchvision.transforms.functional import to_pil_image

def save_val_image(tensor, fname, out_dir="./val_results"):
    os.makedirs(out_dir, exist_ok=True)
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

def validate(model, loader, p_loss_fn, device):
    model.eval()
    psnr_list, ssim_list, p_loss_list = [], [], []

    with torch.no_grad():
        for gray, target, fname in tqdm(loader, desc="Validating"):
            gray, target = gray.to(device), target.to(device)
            pred = model(gray)

            psnr_list.append(psnr(pred, target))
            ssim_list.append(ssim(pred, target))
            p_loss_list.append(p_loss_fn(pred, target).mean().item())

            save_val_image(pred, fname)

    return (
        sum(psnr_list) / len(psnr_list) if psnr_list else 0,
        sum(ssim_list) / len(ssim_list) if ssim_list else 0,
        sum(p_loss_list) / len(p_loss_list) if p_loss_list else 0
    )

def train_one_epoch(model, loader, optimizer, loss_fn, p_loss_fn, device):
    model.train()
    total_loss = 0
    for gray, target, _ in tqdm(loader, desc="Training"):
        gray, target = gray.to(device), target.to(device)
        pred = model(gray)

        rec_loss = loss_fn(pred, target)
        perceptual_loss = p_loss_fn(pred, target).mean()
        loss = rec_loss + 0.1 * perceptual_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader) if len(loader) else 0

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--vendor", type=str, default="vim", choices=["vim","mambairv2","dummy"])
    p.add_argument("--pretrained", action='store_true')
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--lr", type=float, default=1e-4)
    return p.parse_args()

def main():
    args = parse_args()
    distributed = args.local_rank != -1

    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "Distributed:", distributed)

    train_ds = PairedGrayColorDataset(args.data, split='train', size=args.size)
    val_ds   = PairedGrayColorDataset(args.data, split='val', size=args.size, augment=False)

    if distributed:
        train_sampler = DistributedSampler(train_ds)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, sampler=val_sampler, num_workers=2, pin_memory=True)

    model = MambaColorizer(vendor=args.vendor, pretrained=args.pretrained).to(device)
    if distributed:
        model = DDP(model, device_ids=[args.local_rank])

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-6)
    loss_fn = nn.L1Loss()
    p_loss_fn = lpips.LPIPS(net='vgg').to(device)

    os.makedirs("./checkpoints", exist_ok=True)
    best_psnr = 0.0

    for epoch in range(args.epochs):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, p_loss_fn, device)
        avg_psnr, avg_ssim, avg_ploss = validate(model, val_loader, p_loss_fn, device)

        # only rank 0 logs and saves
        if (not distributed) or (dist.get_rank() == 0):
            print(f"\nEpoch [{epoch+1}/{args.epochs}] TrainLoss: {avg_loss:.4f} PSNR: {avg_psnr:.2f} SSIM: {avg_ssim:.4f} LPIPS: {avg_ploss:.4f}")
            log_metrics_to_csv(epoch+1, avg_loss, avg_psnr, avg_ssim, avg_ploss)
            ckpt = {
                "epoch": epoch+1,
                "model_state": model.module.state_dict() if distributed else model.state_dict(),
                "optim_state": optimizer.state_dict()
            }
            torch.save(ckpt, f"./checkpoints/ckpt_epoch{epoch+1}.pth")
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(ckpt, "./checkpoints/best.pth")

if __name__ == "__main__":
    main()
