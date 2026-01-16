import os
import torch
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from src.model import ColorizationModel
from src.loss import MultinomialCrossEntropyLoss
from configs.config import LEARNING_RATE, WEIGHT_DECAY
from src.color_utils import encode_batch_torch, get_cluster_centers

class ColorizationLightningModule(pl.LightningModule):
    def __init__(self, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, class_weights_path=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = ColorizationModel()
        
        # Load cluster centers for GPU encoding
        self.register_buffer("cluster_centers", torch.from_numpy(get_cluster_centers()).float())

        # Load class weights if provided
        weights = None
        if class_weights_path and os.path.exists(class_weights_path):
            import numpy as np
            print(f"Loading class weights from {class_weights_path}")
            w = np.load(class_weights_path)
            weights = torch.from_numpy(w).float()
        
        self.criterion = MultinomialCrossEntropyLoss(weights=weights)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        L, ab = batch # ab is now (B, 2, 112, 112) raw values
        
        # GPU Encoding
        targets = encode_batch_torch(ab, self.cluster_centers) # -> (B, 313, 112, 112)
            
        logits = self(L)
        loss = self.criterion(logits, targets)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        L, ab = batch
        
        # GPU Encoding
        targets = encode_batch_torch(ab, self.cluster_centers)

        logits = self(L)
        loss = self.criterion(logits, targets)
        
        # Metric: Pixel-wise Accuracy (argmax)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1) # (B, H, W)
            targs = torch.argmax(targets, dim=1) # (B, H, W) 
            acc = (preds == targs).float().mean()
            
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # Standard Scheduler: Reduce LR on Plateau
        # Reduces LR when validation loss stops improving
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
