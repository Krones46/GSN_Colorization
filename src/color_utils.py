import os
import numpy as np
import torch
import torch.nn.functional as F
from skimage import color
from sklearn.neighbors import NearestNeighbors
from configs.config import CLUSTER_CENTERS_PATH, T_ANNEAL

"""
Color space utility functions for the colorization project.
Includes RGB<->Lab conversion, soft encoding, and annealed mean decoding.
"""

def rgb_to_lab(image_rgb):
    return color.rgb2lab(image_rgb)

def lab_to_rgb(image_lab):
    return color.lab2rgb(image_lab)

def get_cluster_centers():
    if not os.path.exists(CLUSTER_CENTERS_PATH):
        raise FileNotFoundError(f"Missing {CLUSTER_CENTERS_PATH}")
    return np.load(CLUSTER_CENTERS_PATH)

class ColorEncoder:
    def __init__(self, sigma=5.0):
        self.sigma = sigma
        self.cluster_centers = get_cluster_centers() # (313, 2)
        self.nn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
        self.nn.fit(self.cluster_centers)

    def encode_points_to_soft(self, ab_points):
        """
        ab_points: (N, 2) - pixels in ab space
        Returns: (N, 313) - probability distribution (soft labels)
        """
        # 1. Find nearest clusters for each pixel
        dists, indices = self.nn.kneighbors(ab_points) # dists: (N, 5), indices: (N, 5)

        # 2. Calculate weights from Gaussian distribution
        wts = np.exp(-dists**2 / (2 * self.sigma**2))
        
        # 3. Normalize so weights sum to 1
        wts = wts / np.sum(wts, axis=1, keepdims=True)

        # 4. Create sparse matrix with probabilities
        N = ab_points.shape[0]
        num_classes = self.cluster_centers.shape[0]
        
        # Create output (N, 313)
        soft_enc = np.zeros((N, num_classes), dtype=np.float32)
        
        # Fill with values
        rows = np.arange(N)[:, None]
        soft_enc[rows, indices] = wts
        
        return soft_enc

def compute_loss_weights(counts, lambda_val=0.5):
    """
    Calculates class weights (rebalancing) with smoothing.
    """
    prior_prob = counts / np.sum(counts) + 1e-5
    uniform_prob = 1 / counts.shape[0]
    mixed_prob = (1 - lambda_val) * prior_prob + lambda_val * uniform_prob
    weights = 1 / mixed_prob
    # Normalize to mean 1.0
    weights = weights / np.mean(weights) 
    return weights

def decode_annealed_mean(logits, T=T_ANNEAL):
    """
    Decodes the predicted distribution into ab color values using annealed mean.
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.from_numpy(logits)
    if T < 1e-4: T = 1e-4
    
    probs = F.softmax(logits / T, dim=-3) 
    centers = get_cluster_centers()
    centers_tensor = torch.from_numpy(centers).float().to(logits.device)
        
    if logits.ndim == 4:
        probs = probs.permute(0, 2, 3, 1)
        ab = torch.matmul(probs, centers_tensor)
        ab = ab.permute(0, 3, 1, 2)
    else:
        probs = probs.permute(1, 2, 0)
        ab = torch.matmul(probs, centers_tensor)
        ab = ab.permute(2, 0, 1)
    return ab