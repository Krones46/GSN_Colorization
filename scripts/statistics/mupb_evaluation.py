import os
import sys
import json
import random
import time
import warnings
from collections import defaultdict
import glob

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

"""
Statistical evaluation of the colorization model.
Calculates per-class metrics and generates distribution plots (boxplots, violin plots).
"""

# Path configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    IMAGE_SIZE, DEVICE, CHECKPOINTS_DIR, NUM_COLOR_CLASSES,
    T_ANNEAL, BASE_DIR, DATA_DIR, RAW_DATA_DIR
)
from src.model import ColorizationModel
from src.color_utils import decode_annealed_mean, rgb_to_lab

warnings.filterwarnings("ignore")

# Image limit per class for analysis
IMAGES_PER_CLASS = 50 


def load_synset_mapping():
    # Initialize candidate list for mapping file
    candidates = [
        os.path.join(BASE_DIR, "imagenet_class_index.json"),
        os.path.join(DATA_DIR, "imagenet_class_index.json")
    ]
    mapping = {}
    
    # Attempt to load json file
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                for _, (wnid, name) in data.items():
                    mapping[wnid] = name
                return mapping
            except Exception: pass
    return mapping

def calculate_pixel_accuracy(ab_pred, ab_gt, threshold=5.0):
    # Calculate Euclidean error
    diff = np.sqrt(np.sum((ab_pred - ab_gt) ** 2, axis=2))
    # Return mean pixel count below threshold
    return float((diff <= threshold).mean())

def get_data_source_dir():
    # Get path to validation folder
    val_dir = os.path.join(os.path.dirname(RAW_DATA_DIR), "val")
    
    # Check if validation folder exists and has subfolders
    if os.path.exists(val_dir):
        subdirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
        if len(subdirs) > 0:
            print(f"[INFO] Validation set found.")
            return val_dir
        else:
            print(f"[INFO] Validation set has wrong structure.")
    else:
        print(f"[INFO] No validation folder.")

    print(f"[INFO] Using training set.")
    return RAW_DATA_DIR

def get_all_images_grouped(source_dir, limit_per_class=50):
    if not os.path.exists(source_dir):
        print(f"[ERROR] Folder does not exist: {source_dir}")
        return {}

    print(f"Scanning directories in: {source_dir} ...")
    
    # Get class folder list
    class_folders = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    if not class_folders:
        print("[ERROR] No class folders.")
        return {}
    
    dataset = {}
    total_images = 0
    
    # Iterate classes and collect images
    for i, cls_id in enumerate(class_folders):
        # Show progress every 100 classes
        if i % 100 == 0:
            print(f"   Indexing: {i}/{len(class_folders)}...", end='\r')
            
        cls_path = os.path.join(source_dir, cls_id)
        # Searching for image files
        files = glob.glob(os.path.join(cls_path, "*.JPEG")) + \
                glob.glob(os.path.join(cls_path, "*.jpg")) + \
                glob.glob(os.path.join(cls_path, "*.png"))
        
        if files:
            # Select image sample
            if len(files) > limit_per_class:
                selected = random.sample(files, limit_per_class)
            else:
                selected = files
                
            dataset[cls_id] = selected
            total_images += len(selected)
            
    print(f"\nLoaded {total_images} images from {len(dataset)} classes.")
    return dataset

def main():
    print("Start statistical analysis")

    # Choose data source
    source_dir = get_data_source_dir()
    
    # Get data
    dataset = get_all_images_grouped(source_dir, IMAGES_PER_CLASS)

    if not dataset:
        print("No data.")
        return

    # Load model
    ckpt_path = os.path.join(CHECKPOINTS_DIR, "checkpoint_best.pth.tar")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CHECKPOINTS_DIR, "checkpoint_last.pth.tar")
        
    print(f"Model: {os.path.basename(ckpt_path)}")
    model = ColorizationModel(num_classes=NUM_COLOR_CLASSES).to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    class_scores = defaultdict(list)
    synset_map = load_synset_mapping()
    
    # Prepare list for processing
    flat_list = []
    for cls_id, paths in dataset.items():
        for p in paths:
            flat_list.append((p, cls_id))
            
    # Shuffle list
    random.shuffle(flat_list)
    
    print(f"Processing {len(flat_list)} images...")
    start_time = time.time()

    # Main processing loop
    for idx, (img_path, cls_id) in enumerate(flat_list):
        if idx % 100 == 0 and idx > 0:
            # Calculate estimated completion time
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = (len(flat_list) - idx) * avg_time
            print(f"   Progress: {idx}/{len(flat_list)} - ETA: {remaining/60:.1f} min", end='\r')
            
        try:
            # Load image
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Resize
            img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
            # Normalization
            img_float = img_resized.astype(np.float32) / 255.0 
            
            # Convert to LAB
            lab_gt = rgb_to_lab(img_float) 
            L_gt = lab_gt[:, :, 0]
            ab_gt = lab_gt[:, :, 1:]
            
            # Prepare input tensor
            L_tensor = torch.from_numpy(L_gt).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # Prediction
            with torch.no_grad():
                logits = model(L_tensor)
                ab_decoded = decode_annealed_mean(logits, T=T_ANNEAL)
                ab_pred = ab_decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Calculate metric
            acc = calculate_pixel_accuracy(ab_pred, ab_gt, threshold=5.0)
            
            class_scores[cls_id].append(acc)

        except Exception:
            continue

    duration = time.time() - start_time
    print(f"\nTime: {duration:.1f}s")
    
    print("Aggregating results...")
    
    class_means = []
    for cls_id, scores in class_scores.items():
        if not scores: continue
        mean_score = np.mean(scores)
        class_means.append((cls_id, mean_score))
        
    # Sort classes
    class_means.sort(key=lambda x: x[1], reverse=True)
    
    # Select best and worst for plot
    if len(class_means) >= 30:
        top_15 = class_means[:15]
        bottom_15 = class_means[-15:]
    else:
        mid = len(class_means) // 2
        top_15 = class_means[:mid]
        bottom_15 = class_means[mid:]

    selected_classes = top_15 + bottom_15
    
    plot_data = []
    for cls_id, _ in selected_classes:
        scores = class_scores[cls_id]
        
        cls_name = synset_map.get(cls_id, cls_id)
        # Shorten name
        cls_name_short = cls_name.split(',')[0][:15]
        
        group = "Top 15" if (cls_id, _) in top_15 else "Bottom 15"
        
        for s in scores:
            plot_data.append({
                "Class": cls_name_short, 
                "Pixel Accuracy": s,
                "Group": group
            })
            
    df = pd.DataFrame(plot_data)
    
    if df.empty:
        print("No data for plot.")
        return

    # Establish order on plot
    order = [synset_map.get(cid, cid).split(',')[0][:15] for cid, _ in selected_classes]
    
    seen = set()
    order = [x for x in order if not (x in seen or seen.add(x))]

    # Draw Boxplot
    plt.figure(figsize=(18, 8))
    sns.boxplot(x="Class", y="Pixel Accuracy", hue="Group", data=df, order=order, dodge=False, palette="coolwarm")
    plt.title(f"Pixel Accuracy Distribution")
    plt.ylabel("Pixel Accuracy")
    plt.xlabel("")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "analysis_boxplot_full.png"))
    print(f"Saved boxplot")

    # Draw Violin Plot
    plt.figure(figsize=(18, 8))
    sns.violinplot(x="Class", y="Pixel Accuracy", hue="Group", data=df, order=order, dodge=False, palette="coolwarm", inner="quartile")
    plt.title(f"Distribution Density (Violin Plot)")
    plt.ylabel("Pixel Accuracy")
    plt.xlabel("")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "analysis_violinplot_full.png"))
    print(f"Saved violin plot")
    
    # Save csv
    full_stats = []
    for cls_id, mean_val in class_means:
        full_stats.append({
            "Class ID": cls_id,
            "Name": synset_map.get(cls_id, cls_id),
            "Mean Acc": mean_val,
            "Num Images": len(class_scores[cls_id])
        })
    pd.DataFrame(full_stats).to_csv(os.path.join(BASE_DIR, "full_validation_stats.csv"), index=False)
    print("Saved CSV with results.")

if __name__ == "__main__":
    main()