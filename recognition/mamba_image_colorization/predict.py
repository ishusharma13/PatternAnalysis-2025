# predict.py (updated)
import torch
from modules import MambaColorizer
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import argparse
import os

# Convert PIL image to tensor
def pil_to_tensor(img):
    return to_tensor(img)

# Single image prediction
def run_single(model, img_path, out_path, size=256, device='cuda'):
    img = Image.open(img_path).convert('RGB')
    gray = img.convert('L').resize((size, size))
    gray_t = pil_to_tensor(gray).unsqueeze(0).to(device)

    model.to(device).eval()
    with torch.no_grad():
        out = model(gray_t)

    out_img = to_pil_image(out.squeeze(0).clamp(0, 1))
    os.makedirs(out_path, exist_ok=True)
    save_path = os.path.join(out_path, os.path.basename(img_path))
    out_img.save(save_path)
    print(f"✅ Saved colorized image to: {save_path}")

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="path to trained model checkpoint")
    parser.add_argument("--input", required=True, help="path to grayscale image")
    parser.add_argument("--out", default="./results", help="output directory")
    parser.add_argument("--size", type=int, default=256, help="resize image")
    args = parser.parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = MambaColorizer()
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt))  # handle full checkpoint or raw state_dict

    # Run prediction
    run_single(model, args.input, args.out, args.size, device)
