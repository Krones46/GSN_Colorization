import sys
import os
import numpy as np
from tqdm import tqdm

"""
Script to recompute class weights for loss balancing from processed shards.
Optimized to run quickly by sampling a subset of shards.
"""

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    PROCESSED_DIR, CLASS_WEIGHTS_PATH, NUM_COLOR_CLASSES, 
    CLUSTER_CENTERS_PATH, CLASS_REBAL_LAMBDA
)
from src.color_utils import compute_loss_weights

def main():
    print(f"Fast Weight Recomputation (Lambda={CLASS_REBAL_LAMBDA})")
    
    # 1. Get shard list
    files = [f for f in os.listdir(PROCESSED_DIR) if f.startswith("shard_") and f.endswith(".npz")]
    
    if not files:
        print("ERROR: No shards found in processed folder!")
        return

    # Load cluster centers
    if os.path.exists(CLUSTER_CENTERS_PATH):
        centers = np.load(CLUSTER_CENTERS_PATH).astype(np.float32)
    else:
        print("Warning: Missing cluster_centers.npy")
        centers = None

    total_counts = np.zeros(NUM_COLOR_CLASSES, dtype=np.int64)
    

    # 50 shards (100k images) will give sufficiently accurate statistics in 2 minutes.
    files_to_scan = files[:min(len(files), 50)] 
    
    print(f"Scanning sample of {len(files_to_scan)} shards")

    for f in tqdm(files_to_scan, desc="Counting colors"):
        path = os.path.join(PROCESSED_DIR, f)
        try:
            with np.load(path) as data:
                if 'ab' in data:
                    ab = data['ab'] # (N, H, W, 2)
                    
                    # Sampling for speed (take every 16th pixel)
                    ab_sub = ab[:, ::16, ::16, :] 
                    ab_flat = ab_sub.reshape(-1, 2)

                    # Fast Hard Encoding on the fly (only for statistics!)
                    dists = np.sum((ab_flat[:, None, :] - centers[None, :, :]) ** 2, axis=2)
                    indices = np.argmin(dists, axis=1)
                    
                    unique, counts = np.unique(indices, return_counts=True)
                    total_counts[unique] += counts
                else:
                    print(f"Skipped {f}: missing 'ab' key")

        except Exception as e:
            print(f"Error in file {f}: {e}")
            continue
        
    print("Calculating weights")
    # Using parameter from CONFIG (0.1), not hardcoded
    # This keeps code organized.
    weights = compute_loss_weights(total_counts, lambda_val=CLASS_REBAL_LAMBDA) 
    
    np.save(CLASS_WEIGHTS_PATH, weights)
    print(f"Saved new weights to: {CLASS_WEIGHTS_PATH}")
    print(f"Used Lambda: {CLASS_REBAL_LAMBDA}")
    print(f"Max weight (rare): {weights.max():.2f}")
    print(f"Min weight (background): {weights.min():.2f}")

if __name__ == "__main__":
    main()