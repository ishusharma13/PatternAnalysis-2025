import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import PairedGrayColorDataset
from modules import MambaColorizer
import os

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for gray, target, _ in tqdm(loader, desc="Training"):
        gray, target = gray.to(device), target.to(device)
        pred = model(gray)
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    data_path = "./data"  # update path later
    epochs = 2
    batch_size = 4
    lr = 1e-4
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_ds = PairedGrayColorDataset(data_path, split='train', size=128)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = MambaColorizer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(save_dir, f"epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print("Saved:", ckpt_path)

if __name__ == "__main__":
    main()
