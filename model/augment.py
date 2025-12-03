'''
Data Augmentation Script for YOLO Polygon Segmentation

This script performs data augmentation on a YOLO-format dataset with polygon segmentation labels.
I built this to accomodate the incredibly limited dataset we were given for all views of the goats
This scipt takes the original yolo training images and labels, applies a series of augmentations using Albumentations,
and saves the augmented images and updated polygon labels in YOLO format.
ONLY augments the training set - validation and test sets are copied as-is.

Some augmentations include:
- Horizontal flips
- Small rotations
- Scaling
- Brightness/contrast adjustments
- Gaussian noise

'''

import os
import cv2
import random
import numpy as np
import shutil
from pathlib import Path
import albumentations as A


DATASET_ROOT_DIR = "/Users/ethantenclay/Desktop/goat_seg_side-2" # <--------------- RENAME THIS, I'm using my roboflow dataset downlaod directly 
OUTPUT_ROOT_DIR = "/Users/ethantenclay/Desktop/goatdev/model/side/aug_output"                           # already split into train/valid/test

AUG_PER_IMG = 20 # Generate 20 variants for each training image
SEED = 42


random.seed(SEED)

# Albumentations transform (mask-safe)
transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.0),
        A.ShiftScaleRotate(
            shift_limit=0.02, 
            scale_limit=0.04,
            rotate_limit=3, 
            border_mode=cv2.BORDER_CONSTANT,
            p=0.6
        ),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(p=0.2),
    ]
)

# Load YOLO polygon file (Handles multiple objects and auto-repairs corruption)
# (functions kept the same as the final version)

def load_polygon(label_path):
    all_polygons = []
    
    with open(label_path, "r") as f:
        parts = f.read().strip().replace('\n', ' ').split()

    if not parts:
        return []

    current_cls = -1
    current_coords = []
    
    for part in parts:
        try:
            cls_candidate = int(part)
            
            if current_cls != -1 and current_coords and 0 <= cls_candidate < 1000:
                if len(current_coords) % 2 != 0:
                    print(f"Auto-Repair: Odd coordinates ({len(current_coords)}) for class {current_cls} in {label_path}. Deleting last coordinate.")
                    current_coords.pop()
                
                if current_coords:
                    pts = []
                    for i in range(0, len(current_coords), 2):
                        pts.append([current_coords[i], current_coords[i+1]])
                    all_polygons.append((current_cls, pts))
                    
                current_cls = cls_candidate
                current_coords = []
            elif current_cls == -1 and 0 <= cls_candidate < 1000:
                current_cls = cls_candidate
                
            else:
                coord = float(part)
                current_coords.append(coord)
                
        except ValueError:
            coord = float(part)
            current_coords.append(coord)


    if current_coords:
        if len(current_coords) % 2 != 0:
            print(f"Auto-Repair: Odd coordinates ({len(current_coords)}) for the final class {current_cls} in {label_path}. Deleting last coordinate.")
            current_coords.pop()
        
        if current_coords:
            pts = []
            for i in range(0, len(current_coords), 2):
                pts.append([current_coords[i], current_coords[i+1]])
            all_polygons.append((current_cls, pts))

    return all_polygons

def polygon_to_mask(points, img_w, img_h):
    pts_abs = []
    for x, y in points:
        pts_abs.append([int(x * img_w), int(y * img_h)])
    pts_abs = np.array([pts_abs], dtype=np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, pts_abs, 255)
    return mask

def mask_to_polygon(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    cnt = max(cnts, key=cv2.contourArea)
    poly = []
    for p in cnt.reshape(-1, 2):
        x_norm = p[0] / mask.shape[1]
        y_norm = p[1] / mask.shape[0]
        poly.append([x_norm, y_norm])
    return poly

def save_polygon(path, cls, poly):
    with open(path, "a") as f: 
        flat = []
        for x, y in poly:
            flat.append(f"{x:.6f}")
            flat.append(f"{y:.6f}")
        f.write(str(cls) + " " + " ".join(flat) + "\n")


# Copies original files for val/test splits

def copy_original_split(split_name, input_root, output_root):
    """Copies original images and labels for non-augmented splits (val/test)."""
    input_images = input_root / split_name / "images"
    input_labels = input_root / split_name / "labels"
    output_images = output_root / split_name / "images"
    output_labels = output_root / split_name / "labels"

    # 1. Ensure output directories exist
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    # 2. Copy files
    copied_count = 0
    # Copy images
    for img_path in input_images.glob("*.jpg"):
        shutil.copy2(img_path, output_images / img_path.name)
        copied_count += 1
    # Copy labels
    for lbl_path in input_labels.glob("*.txt"):
        shutil.copy2(lbl_path, output_labels / lbl_path.name)

    print(f"  ✅ Copied {copied_count} files for {split_name} (untouched).")



# Main (Handles all 3 splits)

def main():
    input_root = Path(DATASET_ROOT_DIR)
    output_root = Path(OUTPUT_ROOT_DIR)
    
    SPLITS = ["train", "valid", "test"]

    print(f"Starting data preparation from '{DATASET_ROOT_DIR}' to '{OUTPUT_ROOT_DIR}'...")

    for split in SPLITS:
        
        # Non-training splits: simply copy the originals
        if split != "train":
            print(f"\n--- Processing {split.upper()} Data (Copying) ---")
            copy_original_split(split, input_root, output_root)
            continue
            
        # Training split: perform augmentation
        print(f"\n--- Processing {split.upper()} Data (Augmenting) ---")

        input_base_dir = input_root / split
        out_images_dir = output_root / split / "images"
        out_labels_dir = output_root / split / "labels"
        
        # Ensure output directories exist for the training split
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        search_path = input_base_dir / "images"
        imgs = sorted(search_path.glob("*.jpg"))

        print(f"Found {len(imgs)} original training images to augment.")
        
        # --- Augmentation Logic ---
        for img_path in imgs:
            label_path = input_base_dir / "labels" / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                print(f"No label for {img_path.name}, skipping.")
                continue

            all_polygons_data = load_polygon(label_path) 
            
            if not all_polygons_data:
                print(f"No valid polygons found in {label_path}, skipping image.")
                continue 

            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Could not read image {img_path.name}, skipping.")
                continue
                
            h, w = image.shape[:2]
            
            # 1. Prepare data for Albumentations
            original_masks = []
            original_classes = []
            
            for cls, poly in all_polygons_data:
                base_mask = polygon_to_mask(poly, w, h)
                original_masks.append(base_mask)
                original_classes.append(cls)
            
            # 2. Save original training file to output (for safety)
            orig_img_out = out_images_dir / img_path.name
            orig_lbl_out = out_labels_dir / label_path.name
            shutil.copy2(img_path, orig_img_out) # Copy original image
            shutil.copy2(label_path, orig_lbl_out) # Copy original label
            
            # 3. Handle Augmentations
            for i in range(AUG_PER_IMG):
                aug = transform(image=image, masks=original_masks)
                aug_img = aug["image"]
                aug_masks = aug["masks"] 

                # All augmented files go into the 'train' folder
                img_out = out_images_dir / f"{img_path.stem}_aug{i}.jpg"
                lbl_out = out_labels_dir / f"{img_path.stem}_aug{i}.txt"
                
                # Prepare to write all augmented polygons to a single label file
                if lbl_out.exists():
                    os.remove(lbl_out) 

                valid_aug_masks = 0
                
                for cls_index, aug_mask in zip(original_classes, aug_masks):
                    new_poly = mask_to_polygon(aug_mask)
                    
                    if len(new_poly) >= 3:
                        save_polygon(lbl_out, cls_index, new_poly)
                        valid_aug_masks += 1
                
                if valid_aug_masks > 0:
                    cv2.imwrite(str(img_out), aug_img)


    print("\nDataset preparation complete. Check the 'output' directory.")


if __name__ == "__main__":
    main()