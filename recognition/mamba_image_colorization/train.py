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
import argparse
import os
import csv

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
        sum(psnr_list) / len(psnr_list),
        sum(ssim_list) / len(ssim_list),
        sum(p_loss_list) / len(p_loss_list)
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

    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    data_path = "./data"
    epochs = 2
    batch_size = 4
    lr = 1e-4
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    distributed = args.local_rank != -1

    if distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        if dist.get_rank() == 0:
            print("🐍 Using Distributed Training with DDP")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)

    train_ds = PairedGrayColorDataset(data_path, split='train', size=128)
    val_ds   = PairedGrayColorDataset(data_path, split='val', size=128, augment=False)

    if distributed:
        train_sampler = DistributedSampler(train_ds)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler, val_sampler = None, None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, sampler=val_sampler)

    model = MambaColorizer().to(device)
    
    if distributed:
        model = DDP(model, device_ids=[args.local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    p_loss_fn = lpips.LPIPS(net='vgg').to(device)

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, p_loss_fn, device)

        avg_psnr, avg_ssim, avg_ploss = validate(model, val_loader, p_loss_fn, device)

        if not distributed or dist.get_rank() == 0:
            print(f"\n📌 Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}")
            print(f"✅ Val PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | LPIPS: {avg_ploss:.4f}")
            log_metrics_to_csv(epoch+1, avg_loss, avg_psnr, avg_ssim, avg_ploss)
            ckpt_path = os.path.join(save_dir, f"epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print("💾 Saved:", ckpt_path)

if __name__ == "__main__":
    main()
