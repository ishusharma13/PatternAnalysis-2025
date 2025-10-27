import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class GrayscaleColorDataset(Dataset):
    def __init__(self, gray_folder, color_folder, transform=None):
        self.gray_images = sorted([os.path.join(gray_folder, f) for f in os.listdir(gray_folder) if f.lower().endswith(('.jpg','.png'))])
        self.color_images = sorted([os.path.join(color_folder, f) for f in os.listdir(color_folder) if f.lower().endswith(('.jpg','.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.gray_images)

    def __getitem__(self, idx):
        gray_img = Image.open(self.gray_images[idx]).convert("L")
        color_img = Image.open(self.color_images[idx]).convert("RGB")
        
        if self.transform:
            gray_img = self.transform(gray_img)
            color_img = self.transform(color_img)
        
        return gray_img, color_img

# Define transforms (convert to tensor, normalize)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Training dataset and loader
train_dataset = GrayscaleColorDataset("coco_train_final/gray", "coco_train_final/color", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Validation dataset and loader
val_dataset = GrayscaleColorDataset("coco_val_final/gray", "coco_val_final/color", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
