import os
import sys
import numpy as np

"""
Simple script to inspect the generated statistical files (cluster centers, class weights).
"""

# Add path to project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import CLUSTER_CENTERS_PATH, CLASS_WEIGHTS_PATH

def inspect_statistics():
    print(f"Statistics")
    
    # Check cluster centers file
    if os.path.exists(CLUSTER_CENTERS_PATH):
        centers = np.load(CLUSTER_CENTERS_PATH)
        print(f"\nCluster centers:")
        print(f"    Path: {CLUSTER_CENTERS_PATH}")
        print(f"    Table shape: {centers.shape}")
        
        # Display value ranges  
        print(f"    Range 'a': min={centers[:, 0].min():.2f}, max={centers[:, 0].max():.2f}")
        print(f"    Range 'b': min={centers[:, 1].min():.2f}, max={centers[:, 1].max():.2f}")
        
        print(f"    Example 5 centers:")
        print(centers[:5])
    else:
        print(f"Missing file {CLUSTER_CENTERS_PATH}")

    # Check class weights file
    if os.path.exists(CLASS_WEIGHTS_PATH):
        weights = np.load(CLASS_WEIGHTS_PATH)
        print(f"\nClass weights:")
        print(f"    Path: {CLASS_WEIGHTS_PATH}")
        print(f"    Table shape: {weights.shape}")
        
        # Weight statistics
        print(f"    Mean weight: {weights.mean():.4f}")
        print(f"    Min weight: {weights.min():.4f}")
        print(f"    Max weight: {weights.max():.4f}")
    else:
        print(f"Missing file {CLASS_WEIGHTS_PATH}")

if __name__ == "__main__":
    inspect_statistics()