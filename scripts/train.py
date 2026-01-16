import sys
import os
import torch
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lightning_module import ColorizationLightningModule
from src.dataset import ColorizationIterableDataset
from configs.config import PROCESSED_DIR, CLASS_WEIGHTS_PATH, TRAIN_NUM_SHARDS

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    
    # Optimize for Tensor Cores
    torch.set_float32_matmul_precision('medium')
    
    # 1. DataPaths
    data_dir = cfg.data.get("processed_dir", PROCESSED_DIR)
    if not os.path.isabs(data_dir) and not os.path.exists(data_dir):
        # Fallback to config if relative path fails (Hydra changes cwd)
        import hydra.utils
        data_dir = os.path.join(hydra.utils.get_original_cwd(), data_dir)

    print(f"Data Dir: {data_dir}")
    
    # 2. Datasets & Loaders
    limit_shards = cfg.data.get("limit_shards", TRAIN_NUM_SHARDS)
    
    if limit_shards is None:
        print("INFO: Training on FULL dataset (limit_shards=None).")
    else:
        print(f"INFO: Limiting dataset to {limit_shards} shards.")
    
    train_dataset = ColorizationIterableDataset(
        processed_dir=data_dir, 
        split="train", 
        train_split=cfg.data.split_ratio,
        limit_shards=limit_shards
    )
    val_dataset = ColorizationIterableDataset(
        processed_dir=data_dir, 
        split="val", 
        train_split=cfg.data.split_ratio,
        limit_shards=limit_shards
    )

    # Determine persistent_workers based on num_workers
    use_persistent_workers = (cfg.data.num_workers > 0)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
    )

    # 3. Model
    # Helper to resolve weight path
    cw_path = cfg.model.get("class_weights_path", CLASS_WEIGHTS_PATH)
    if not os.path.isabs(cw_path) and not os.path.exists(cw_path):
        import hydra.utils
        cw_path = os.path.join(hydra.utils.get_original_cwd(), cw_path)

    model = ColorizationLightningModule(
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        class_weights_path=cw_path
    )

    # 4. Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='colorization-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=cfg.trainer.log_every_n_steps
    )

    # 6. Train
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
