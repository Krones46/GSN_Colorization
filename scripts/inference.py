
import os
import sys
import json
import random
import time
from collections import defaultdict
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
This script performs inference and evaluation for the colorization model.
It loads a trained model, processes a subset of test images, and calculates 
metrics such as PSNR, RMSE, AuC, and Colorfulness ratio.
"""

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    IMAGE_SIZE, DEVICE, CHECKPOINTS_DIR, NUM_COLOR_CLASSES,
    TEST_SUBSET_PATH, T_ANNEAL, BASE_DIR, DATA_DIR
)
from src.model import ColorizationModel
from src.color_utils import decode_annealed_mean, rgb_to_lab, lab_to_rgb

warnings.filterwarnings("ignore", category=DeprecationWarning)

NUM_SAMPLES = 100
ckpt_path = os.path.join(CHECKPOINTS_DIR, "checkpoint_last.pth.tar")
if not os.path.exists(ckpt_path):
    ckpt_path = os.path.join(CHECKPOINTS_DIR, "checkpoint_last.pth.tar")


def calculate_psnr(img_true, img_pred):
    """
    Peak Signal-to-Noise Ratio
    img_true, img_pred: RGB images in [0, 1]
    """
    mse = np.mean((img_true - img_pred) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)

def calculate_accuracy_auc(diff_ab):
    """
    Calculates Area Under Curve (AuC) for error in ab space.
    """
    thresholds = np.linspace(0, 150, 151)
    accs = []
    
    # For each threshold check what % of pixels has lower error
    for t in thresholds:
        acc = np.mean(diff_ab <= t)
        accs.append(acc)
        
    # Normalized area under curve
    auc = np.trapz(accs, thresholds) / 150.0
    return auc, accs

def colorfulness(img_rgb):
    """
    Measures color 'vibrancy'.
    """
    # rg = R - G
    # yb = 0.5 * (R + G) - B
    R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    
    std_root = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
    mean_root = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    
    return std_root + (0.3 * mean_root)


def main():
    if not os.path.exists(TEST_SUBSET_PATH):
        print(f"ERROR: {TEST_SUBSET_PATH} not found")
        return

    with open(TEST_SUBSET_PATH, "r") as f:
        all_test_paths = json.load(f)

    eval_paths = random.sample(all_test_paths, min(NUM_SAMPLES, len(all_test_paths)))
    
    print(f"Model: {os.path.basename(ckpt_path)}")
    model = ColorizationModel(num_classes=NUM_COLOR_CLASSES).to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    metrics = defaultdict(list)    

    vis_dir = os.path.join(BASE_DIR, "evaluation_results")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"Processing {len(eval_paths)} images")
    
    start_time = time.time()
    
    for idx, img_path in enumerate(eval_paths):
        try:

            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            img_resized = cv2.resize(img_rgb_orig, (IMAGE_SIZE, IMAGE_SIZE))
            img_float = img_resized.astype(np.float32) / 255.0 # (224,224,3) 0..1
            
            lab_gt = rgb_to_lab(img_float) 
            L_gt = lab_gt[:, :, 0]
            ab_gt = lab_gt[:, :, 1:]
            
            L_input = L_gt / 100.0
            L_tensor = torch.from_numpy(L_input).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = model(L_tensor)
                ab_decoded = decode_annealed_mean(logits, T=T_ANNEAL) # (1, 2, H, W)
                ab_pred = ab_decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            if ab_pred.shape[:2] != L_gt.shape[:2]:
                ab_pred = cv2.resize(ab_pred, (L_gt.shape[1], L_gt.shape[0]))


            # Combine ORIGINAL L with predicted ab
            lab_pred = np.dstack((L_gt, ab_pred))
            img_rgb_pred = lab_to_rgb(lab_pred).clip(0, 1) 
            

            # 1. RMSE (Root Mean Square Error) in ab channel
            mse_ab = np.mean((ab_pred - ab_gt) ** 2)
            rmse = np.sqrt(mse_ab)
            
            # 2. PSNR (on RGB)
            psnr = calculate_psnr(img_float, img_rgb_pred)
            
            # 3. AuC Accuracy 
            dist_map = np.sqrt(np.sum((ab_pred - ab_gt) ** 2, axis=2))
            auc, accs = calculate_accuracy_auc(dist_map)
            acc5 = accs[5]   # Threshold = 5%
            
            # 4. Colorfulness 
            cf_pred = colorfulness(img_rgb_pred)
            cf_gt = colorfulness(img_float)
            cf_ratio = cf_pred / (cf_gt + 1e-6)

            # Saving
            metrics["rmse"].append(rmse)
            metrics["psnr"].append(psnr)
            metrics["auc"].append(auc)
            metrics["acc5"].append(acc5)
            metrics["cf_ratio"].append(cf_ratio)
            metrics["cf_pred"].append(cf_pred)
            metrics["cf_gt"].append(cf_gt)

        except Exception as e:
            print(f"Error on {img_path}: {e}")
            continue


    duration = time.time() - start_time
    print(f"\n Results ({duration:.1f}s)")
    print(f"Number of images: {len(metrics['rmse'])}")
    
    # Averages
    mean_rmse = np.mean(metrics["rmse"])
    mean_psnr = np.mean(metrics["psnr"])
    mean_auc = np.mean(metrics["auc"])
    mean_acc5 = np.mean(metrics["acc5"])
    
    # Colorfulness Ratio 
    # Mean of Ratios explodes when denominator (GT) is close to 0 (black and white image).
    mean_cf_pred = np.mean(metrics["cf_pred"])
    mean_cf_gt = np.mean(metrics["cf_gt"])
    global_cf_ratio = mean_cf_pred / (mean_cf_gt + 1e-6)
    
    print("\nMetrics:")
    print(f"Raw Accuracy:")
    print(f"AuC (Overall Quality): {mean_auc:.4f} (higher is better, max ~0.9)")
    print(f"Acc@5 (Very precise): {mean_acc5*100:.2f}%")
    
    print(f"Errors and Noise:")
    print(f"RMSE (ab channel): {mean_rmse:.2f} (lower is better, typically 10-15)")
    print(f"PSNR (RGB): {mean_psnr:.2f} dB (higher is better, typically 20-25)")
    
    print(f"Aesthetics:")
    print(f"Colorfulness Ratio: {global_cf_ratio:.2f}")
    print(f"Avg Pred Colorfulness: {mean_cf_pred:.4f}")
    print(f"Avg GT Colorfulness:   {mean_cf_gt:.4f}")


if __name__ == "__main__":
    main()