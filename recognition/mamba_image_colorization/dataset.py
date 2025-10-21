import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

def pil_to_tensor(img: Image.Image):
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    arr = torch.from_numpy(arr).permute(2,0,1)
    return arr

class PairedGrayColorDataset(Dataset):
    def __init__(self, root_dir, split='train', size=256, augment=True):
        super().__init__()
        self.root = os.path.join(root_dir, split)
        if not os.path.exists(self.root):
            raise ValueError(f"{self.root} does not exist")
        self.paths = [os.path.join(self.root,f) for f in os.listdir(self.root)
                      if f.lower().endswith(('.png','.jpg','.jpeg'))]
        self.size = size
        self.augment = augment
        self.base_transforms = T.Compose([
            T.Resize((size,size))
        ])
        self.aug_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        if self.augment:
            img = self.aug_transforms(img)
        img = self.base_transforms(img)
        target = pil_to_tensor(img)  # 3,H,W
        gray = img.convert('L')
        gray = pil_to_tensor(gray)   # 1,H,W
        return gray, target, os.path.basename(p)
