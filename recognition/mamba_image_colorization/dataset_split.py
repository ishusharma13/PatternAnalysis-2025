import os, random, shutil

source_folder = "coco_train_subset"
train_folder = "coco_train_final"
val_folder = "coco_val_final"

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

all_images = os.listdir(source_folder)
random.shuffle(all_images)

# Split 200 for validation, rest for training
val_size = 200
val_images = all_images[:val_size]
train_images = all_images[val_size:]

# Move/copy images
for img in train_images:
    shutil.copy(os.path.join(source_folder, img), os.path.join(train_folder, img))

for img in val_images:
    shutil.copy(os.path.join(source_folder, img), os.path.join(val_folder, img))

print(f"Training images: {len(train_images)}, Validation images: {len(val_images)}")
