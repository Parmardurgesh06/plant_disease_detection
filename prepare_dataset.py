
import os
import random
import shutil

SOURCE_DIR = "/Users/parmardurgesh/Desktop/Tomato-leaf-disease-dataset"
DEST_DIR = "dataset"

CLASSES = ["healthy", "Early_blight", "Late_blight"]

# Split: 70% train, 15% validation, 15% test
random.seed(42)

for class_name in CLASSES:
    source_class = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(source_class):
        print("Folder not found:", source_class)
        continue

    images = [
        name for name in os.listdir(source_class)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    print(f"\n{class_name}: {total} images")

    for split, filenames in splits.items():
        target_folder = os.path.join(
            DEST_DIR, split, class_name
        )
        os.makedirs(target_folder, exist_ok=True)

        for filename in filenames:
            source = os.path.join(source_class, filename)
            destination = os.path.join(target_folder, filename)

            shutil.copy2(source, destination)

        print(f"{split}: {len(filenames)} images")

print("\nDataset preparation complete!")
