from PIL import Image
import os

def prepare_grayscale_pairs(source_folder):
    gray_folder = os.path.join(source_folder, "gray")
    color_folder = os.path.join(source_folder, "color")
    os.makedirs(gray_folder, exist_ok=True)
    os.makedirs(color_folder, exist_ok=True)

    for img_file in os.listdir(source_folder):
        img_path = os.path.join(source_folder, img_file)
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = Image.open(img_path).convert("RGB")
        img.save(os.path.join(color_folder, img_file))      # color target
        gray_img = img.convert("L")
        gray_img.save(os.path.join(gray_folder, img_file))  # grayscale input

# Run for training and validation
prepare_grayscale_pairs("coco_train_final")
prepare_grayscale_pairs("coco_val_final")

print("Grayscale → color pairs ready!")
