import sys
import os
import json
import numpy as np
from tqdm import tqdm
from skimage import io, transform, color
from sklearn.cluster import MiniBatchKMeans
from concurrent.futures import ProcessPoolExecutor
import functools

"""
Data preparation script.
Scans images, computes cluster centers (anchors) for quantization,
and processes images into shards for efficient streaming training.
"""

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    RAW_DATA_DIR, PROCESSED_DIR, CLUSTER_CENTERS_PATH,
    CLASS_WEIGHTS_PATH, IMAGE_SIZE, NUM_COLOR_CLASSES,
    DATA_SUBSET_SIZE
)
from src.color_utils import compute_loss_weights


def get_subset_image_paths(root_dir, max_images=None):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_paths = []

    print(f"Scanning {root_dir} for images...")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))

    image_paths = np.array(image_paths)
    num_found = len(image_paths)
    print(f"Found {num_found} candidate image files.")

    # If max_images is None -> take everything (e.g. full ImageNet)
    if max_images is not None and num_found > max_images:
        print(f"Taking subset of {max_images} images from found {num_found}.")
        # random sample to make dataset reasonably representative
        idx = np.random.choice(num_found, size=max_images, replace=False)
        image_paths = image_paths[idx]

    # shuffle order (for better class mix in shards)
    np.random.shuffle(image_paths)
    return list(image_paths)


def compute_cluster_centers(image_paths, num_clusters=NUM_COLOR_CLASSES):
    """
    Calculates class centers in ab space on a subset of images.
    Uses MiniBatchKMeans + limits pixel count to avoid killing RAM.
    """
    print("Computing cluster centers (MiniBatchKMeans)...")

    # How many images to use for clustering
    max_images_for_kmeans = 2000
    subset_paths = image_paths[:min(len(image_paths), max_images_for_kmeans)]

    # maximum number of pixels for clustering
    MAX_PIXELS = 1_000_000

    ab_pixels_list = []

    for img_path in tqdm(subset_paths, desc="Loading for clustering"):
        try:
            img = io.imread(img_path)
            if img.ndim == 2:
                img = color.gray2rgb(img)
            elif img.shape[2] == 4:
                img = color.rgba2rgb(img)

            img = transform.resize(img, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True)
            lab = color.rgb2lab(img)
            ab = lab[:, :, 1:]

            ab_flat = ab.reshape(-1, 2)

            # Take only a fraction of pixels from the image (here 5%)
            n_pixels = ab_flat.shape[0]
            n_sample = max(1, int(n_pixels * 0.05))
            idx = np.random.choice(n_pixels, size=n_sample, replace=False)
            ab_pixels_list.append(ab_flat[idx])

        except Exception:
            # if image fails to load / is corrupted -> skip
            continue

    if not ab_pixels_list:
        raise RuntimeError("Could not load any images for clustering!")

    ab_pixels = np.concatenate(ab_pixels_list, axis=0)
    total_pixels = ab_pixels.shape[0]
    print(f"Collected {total_pixels} ab-pixels for clustering.")

    # if we collected too many points, subsample globally
    if total_pixels > MAX_PIXELS:
        print(f"Subsampling to {MAX_PIXELS} pixels for MiniBatchKMeans")
        idx = np.random.choice(total_pixels, size=MAX_PIXELS, replace=False)
        ab_pixels = ab_pixels[idx]

    print("Running MiniBatchKMeans")
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=10_000,
        n_init=5,
        random_state=42,
        verbose=0
    )
    kmeans.fit(ab_pixels)

    centers = kmeans.cluster_centers_
    os.makedirs(os.path.dirname(CLUSTER_CENTERS_PATH), exist_ok=True)
    np.save(CLUSTER_CENTERS_PATH, centers)
    print(f"Saved cluster centers to {CLUSTER_CENTERS_PATH}")
    return centers


# Worker function for multiprocessing
def process_single_image(img_path, centers, image_size):
    """
    Processes a single image
    Returns: (L_norm, ab_raw, counts) or None on failure.
    """
    try:
        img = io.imread(img_path)
        if img.ndim == 2:
            img = color.gray2rgb(img)
        elif img.shape[2] == 4:
            img = color.rgba2rgb(img)

        # Resize
        img_resized = transform.resize(img, (image_size, image_size), anti_aliasing=True)

        # LAB
        lab = color.rgb2lab(img_resized)
        L = lab[:, :, 0]  # [0, 100]
        ab = lab[:, :, 1:] # [-128, 128]

        # Normalize L to [0, 1]
        L_norm = L / 100.0
        
        # Save raw AB as float32
        ab_raw = ab.astype(np.float32)

        # Weight Statistics
        h, w = ab.shape[:2]
        ab_flat = ab.reshape(-1, 2)  # (H*W, 2)

        # Calculate distances to centers
        dists = np.sum((ab_flat[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        indices_flat = np.argmin(dists, axis=1)
        
        # Count classes for weights
        unique, counts = np.unique(indices_flat, return_counts=True)
        dense_counts = np.zeros(len(centers), dtype=np.int64)
        dense_counts[unique] += counts

        return L_norm, ab_raw, dense_counts

    except Exception:
        return None


def process_data():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Get Paths
    image_paths = get_subset_image_paths(
        RAW_DATA_DIR,
        max_images=DATA_SUBSET_SIZE
    )
    print(f"Using {len(image_paths)} images for ETL.")

    if not image_paths:
        print("No images found. Check RAW_DATA_DIR in config.")
        return

    # 2. Check/Compute Cluster Centers
    if not os.path.exists(CLUSTER_CENTERS_PATH):
        centers = compute_cluster_centers(image_paths)
    else:
        print("Loading existing cluster centers")
        centers = np.load(CLUSTER_CENTERS_PATH)
    # for safety
    centers = centers.astype(np.float32)

    # 3. Process Images in Parallel (STREAMING)
    SHARD_SIZE = 2048
    current_shard_L = []
    current_shard_ab = [] # Buffer list name change
    shard_idx = 0
    shard_names = []

    class_counts_global = np.zeros(NUM_COLOR_CLASSES, dtype=np.int64)

    max_workers = 8

    print(f"Starting multiprocessing with {max_workers} workers")

    worker_func = functools.partial(
        process_single_image,
        centers=centers,
        image_size=IMAGE_SIZE
    )

    processed_ok = 0
    processed_failed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(
            executor.map(worker_func, image_paths, chunksize=32),
            total=len(image_paths),
            desc="Processing"
        ):
            if res is None:
                processed_failed += 1
                continue

            processed_ok += 1
            L_norm, ab_data, counts = res

            # Update global stats
            class_counts_global += counts

            # Buffer for shard
            # L_norm: (H, W) -> (1, H, W)
            current_shard_L.append(L_norm[np.newaxis, :, :])
            # ab_data: (H, W, 2)
            current_shard_ab.append(ab_data)

            # Save shard if full
            if len(current_shard_L) >= SHARD_SIZE:
                shard_name = f"shard_{shard_idx}.npz"
                save_path = os.path.join(PROCESSED_DIR, shard_name)
                
                np.savez_compressed(
                    save_path,
                    L=np.array(current_shard_L, dtype=np.float32),
                    ab=np.array(current_shard_ab, dtype=np.float32) 
                )
                shard_names.append(shard_name)

                current_shard_L = []
                current_shard_ab = []
                shard_idx += 1

    print(f"Processed OK: {processed_ok}, failed: {processed_failed}")

    # Save last shard
    if current_shard_L:
        shard_name = f"shard_{shard_idx}.npz"
        save_path = os.path.join(PROCESSED_DIR, shard_name)
        np.savez_compressed(
            save_path,
            L=np.array(current_shard_L, dtype=np.float32),
            ab=np.array(current_shard_ab, dtype=np.float32)
        )
        shard_names.append(shard_name)
        print(f"Saved final shard: {shard_name}")

    # Save Index
    index_data = {
        "shard_names": shard_names,
        "total_examples": processed_ok
    }
    with open(os.path.join(PROCESSED_DIR, "index.json"), "w") as f:
        json.dump(index_data, f)

    # 4. Compute & Save Weights
    print("Computing class weights")
    weights = compute_loss_weights(class_counts_global)
    np.save(CLASS_WEIGHTS_PATH, weights)
    print(f"Saved class weights to {CLASS_WEIGHTS_PATH}")
    print("ETL process complete")


if __name__ == "__main__":
    process_data()