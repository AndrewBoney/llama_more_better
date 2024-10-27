import lightning as L

import os
import wandb
import torch

from datetime import datetime
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoModelForCausalLM

from llm_more_better.model import RewardModelLM
from llm_more_better.data import get_anthropic_rlhf_data

def train_reward_model(
    model_name="anthropic/claude-3-haiku-20240307",
    batch_size=4,
    max_epochs=10,
    learning_rate=1e-4,
    weight_decay=0.01,
    grad_clip=1.0,
    precision="16-mixed",
    seed=42,
    wandb_project="rlhf-reward-model",
    num_workers=4,
    save_dir="checkpoints",
):
    """
    Train a reward model using PyTorch Lightning
    
    Args:
        model_name (str): Base model to use
        batch_size (int): Batch size for training
        max_epochs (int): Maximum number of epochs to train
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay for AdamW
        grad_clip (float): Gradient clipping value
        precision (str): Training precision
        seed (int): Random seed
        wandb_project (str): W&B project name
        num_workers (int): Number of workers for data loading
        save_dir (str): Directory to save checkpoints
    """
    # Set seeds for reproducibility
    L.seed_everything(seed)
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=f"reward_model_{timestamp}",
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "precision": precision,
            "seed": seed,
        }
    )
    
    # Load base model
    print(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Initialize reward model
    print("Initializing reward model...")
    model = RewardModelLM(
        model=base_model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=max_epochs
    )
    
    # Get data loaders
    print("Loading and processing datasets...")
    train_loader, val_loader, test_loader = get_anthropic_rlhf_data(
        batch_size=batch_size,
        seed=seed,
        model_name=model_name,
    )
    
    # Setup callbacks
    callbacks = [
        # Save best models based on validation loss
        ModelCheckpoint(
            dirpath=save_dir,
            filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        ),
        # Save latest model
        ModelCheckpoint(
            dirpath=save_dir,
            filename="latest-checkpoint",
            every_n_epochs=1,
        ),
        # Monitor learning rate
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=precision,
        gradient_clip_val=grad_clip,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        accumulate_grad_batches=4,  # Effective batch size = batch_size * 4
        log_every_n_steps=10,
        val_check_interval=0.25,  # Validate every 25% of epoch
    )
    
    # Train model
    print("Starting training...")
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    # Test model
    print("Testing model...")
    trainer.test(model=model, dataloaders=test_loader)
    
    # Close wandb run
    wandb.finish()
    
    return model, trainer

def main():
    """Main training function with default hyperparameters"""
    # You can modify these parameters as needed
    trained_model, trainer = train_reward_model(
        model_name="anthropic/claude-3-haiku-20240307",
        batch_size=4,
        max_epochs=10,
        learning_rate=1e-4,
        weight_decay=0.01,
        grad_clip=1.0,
        precision="16-mixed",
        seed=42,
        wandb_project="rlhf-reward-model",
        num_workers=4,
        save_dir="checkpoints",
    )
    
    print("Training completed!")
    
    # Print best validation loss
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    print(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.4f}")

if __name__ == "__main__":
    main()