import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

"""
Main training script for the colorization model.
Handles training loop, validation, checkpointing, and mixed precision training.
"""

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    NUM_WORKERS, USE_AMP, DEVICE, CHECKPOINTS_DIR,  LOG_DIR,
    CLASS_WEIGHTS_PATH, RESUME_FROM
)

from src.model import ColorizationModel
from src.dataset import ColorizationIterableDataset

class AverageMeter:
    # Average meter class.
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0: self.avg = self.sum / self.count

def validate(model, loader, criterion):
    # Validation function.
    model.eval() 
    loss_meter = AverageMeter()
    
    from configs.config import TRAIN_NUM_SHARDS
    
    pbar = tqdm(loader, desc="Validating", leave=False)
    
    with torch.no_grad(): 
        for i, (L, targets) in enumerate(pbar):
            if TRAIN_NUM_SHARDS is not None and TRAIN_NUM_SHARDS < 500 and i > 300:
                break

            L = L.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            
            # Upsample on GPU
            if targets.shape[-1] != 112:
                targets = F.interpolate(targets, size=(112, 112), mode='nearest')
            
            logits = model(L)
            loss = criterion(logits, targets)
            loss_meter.update(loss.item(), L.size(0))
            
    return loss_meter.avg

def train():
    # 1. Setup
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"AMP: {USE_AMP}")

    # 2. Data
    print("Initializing Datasets")
    train_dataset = ColorizationIterableDataset(split="train")
    val_dataset = ColorizationIterableDataset(split="val") 
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=max(1, NUM_WORKERS // 2), 
        pin_memory=False,
    )

    # 3. Model
    model = ColorizationModel().to(DEVICE)

    # 4. Loss
    if os.path.exists(CLASS_WEIGHTS_PATH):
        print(f"Loading class weights from {CLASS_WEIGHTS_PATH}")
        weights = np.load(CLASS_WEIGHTS_PATH)
        weights = torch.from_numpy(weights).float().to(DEVICE)
    else:
        print("Class weights not found")
        weights = torch.ones(NUM_COLOR_CLASSES).to(DEVICE)

    class MultinomialCrossEntropyLoss(nn.Module):
        def __init__(self, weights=None):
            super().__init__()
            self.weights = weights
            
        def forward(self, logits, targets):
            log_probs = F.log_softmax(logits, dim=1)
            loss_map = - (targets * log_probs)
            if self.weights is not None:
                w_broadcast = self.weights.view(1, -1, 1, 1)
                loss_map = loss_map * w_broadcast
            loss = loss_map.sum(dim=1).mean() 
            return loss

    criterion = MultinomialCrossEntropyLoss(weights=weights)

    # 5. Optimizer & Scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )
    
    scaler = torch.amp.GradScaler("cuda") if USE_AMP and DEVICE == "cuda" else None

    # Resuming
    start_epoch = 1
    best_val_loss = float('inf') 
    
    if RESUME_FROM and os.path.isfile(RESUME_FROM):
        print(f"Resuming from: {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler and "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float('inf'))

    # 6. Training loop
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # Training
        model.train()
        train_loss_meter = AverageMeter()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")

        for L, targets in pbar:
            L = L.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            # Upsample on GPU
            # Model outputs 112x112. Targets should be 112x112.
            if targets.shape[-1] != 112:
                targets = F.interpolate(targets, size=(112, 112), mode='nearest')


            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast("cuda"):
                    logits = model(L)
                    loss = criterion(logits, targets)

                if torch.isnan(loss):
                    print("NaN detected. Skipping batch.")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(L)
                loss = criterion(logits, targets)
                if torch.isnan(loss): continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss_meter.update(loss.item(), L.size(0))
            pbar.set_postfix(loss=train_loss_meter.avg)

        # Validation
        print(f"Validating epoch {epoch}...")
        val_loss = validate(model, val_loader, criterion)
        
        # Scheduler step
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch} Done. "
            f"Train Loss: {train_loss_meter.avg:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"New best model! Loss: {best_val_loss:.4f}")

        save_dict = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "loss": train_loss_meter.avg,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss
        }
        
        torch.save(save_dict, os.path.join(CHECKPOINTS_DIR, f"checkpoint_last.pth.tar"))
        
        if is_best:
            torch.save(save_dict, os.path.join(CHECKPOINTS_DIR, f"checkpoint_best.pth.tar"))

if __name__ == "__main__":
    train()