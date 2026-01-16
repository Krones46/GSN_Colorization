import os
import json
import numpy as np
import torch
from torch.utils.data import IterableDataset
import cv2

from configs.config import PROCESSED_DIR, TRAIN_NUM_SHARDS
from src.color_utils import ColorEncoder

"""
Dataset implementation for the colorization project.
Handles data loading, splitting, and preprocessing.
"""

class ColorizationIterableDataset(IterableDataset):
    """
    Iterable Dataset - streams data by shards.
    """

    def __init__(self, processed_dir: str = PROCESSED_DIR,
                 split: str = "train", train_split: float = 0.9, limit_shards: int = None):
        super().__init__()
        self.processed_dir = processed_dir
        self.split = split
        self.train_split = train_split

        index_path = os.path.join(processed_dir, "index.json")
        if not os.path.exists(index_path):
            raise RuntimeError(
                f"Index file not found at {index_path}"
            )

        with open(index_path, "r") as f:
            index_data = json.load(f)

        all_shard_names = index_data.get("shard_names", [])
        if not all_shard_names:
            raise RuntimeError("No shard_names found in index.json.")

        # Split Train / Val
        total_shards = len(all_shard_names)
        split_point = int(total_shards * train_split)
        
        if split_point == 0 and total_shards > 0:
            split_point = 1 

        if split == "train":
            available_shards = all_shard_names[:split_point]
            # Priority: limit_shards arg > TRAIN_NUM_SHARDS config
            limit = None
            if limit_shards is not None:
                limit = limit_shards
            elif TRAIN_NUM_SHARDS is not None:
                limit = TRAIN_NUM_SHARDS
            
            if limit is not None:
                limit = min(limit, len(available_shards))
                self.shard_names = available_shards[:limit]
            else:
                self.shard_names = available_shards
        else:
            self.shard_names = all_shard_names[split_point:]
            # Limit validation set size if requested
            if limit_shards is not None:
                limit = min(limit_shards, len(self.shard_names))
                self.shard_names = self.shard_names[:limit]

        print(f"[{split}] Initialized IterableDataset with {len(self.shard_names)} shards")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Disable OpenCV multithreading in workers to prevent CPU overload on Windows
        cv2.setNumThreads(0)
        
        # Copy list to avoid modifying original in self
        shards = self.shard_names.copy()

        if worker_info is not None:
            # Distribute work among workers
            per_worker = int(np.ceil(len(shards) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(shards))
            my_shards = shards[iter_start:iter_end]
        else:
            my_shards = shards

        if self.split == "train":
            np.random.shuffle(my_shards)

        for shard_name in my_shards:
            shard_path = os.path.join(self.processed_dir, shard_name)
            
            try:
                with np.load(shard_path) as data:
                    L_chunk = data['L']       # (N, 1, H, W)
                    ab_chunk = data['ab']     # (N, H, W, 2)
                
                num_samples = L_chunk.shape[0]
                idxs = np.arange(num_samples)

                if self.split == "train":
                    np.random.shuffle(idxs)

                for i in idxs:
                    L_img = L_chunk[i]      # (1, H, W)
                    ab_img = ab_chunk[i]    # (H, W, 2)
                    
                    # Augmentation (Flip)
                    if self.split == 'train' and np.random.rand() > 0.5:
                        L_img = np.flip(L_img, axis=-1).copy()
                        ab_img = np.flip(ab_img, axis=1).copy()

                    L_tensor = torch.from_numpy(L_img).float()
                    
                    # Half resolution (112x112)
                    # Use cv2 for speed (ab_img is HxWx2)
                    ab_resized = cv2.resize(ab_img, (112, 112), interpolation=cv2.INTER_LINEAR)
                    
                    # Tensor (2, 112, 112)
                    ab_tensor = torch.from_numpy(ab_resized).float().permute(2, 0, 1)
                    
                    yield L_tensor, ab_tensor

            except Exception as e:
                print(f"Error loading shard {shard_path}: {e}")
                continue