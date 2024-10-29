import lightning as L

import os
import argparse
import wandb
import torch

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from datetime import datetime

from llm_more_better.model import RewardModelLM
from llm_more_better.data import get_anthropic_rlhf_data

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train a reward model for RLHF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Base model to use"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW"
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping value"
    )
    
    parser.add_argument(
        "--use_lora", 
        type=bool,
        default = True, 
        help="Use LoRA for fine-tuning"
    )

    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=16, 
        help="LoRA attention dimension"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=32, 
        help="LoRA alpha scaling"
    )
    parser.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.1, 
        help="LoRA dropout"
    )

    # Training setup
    parser.add_argument(
        "--precision",
        type=str,
        choices=["32", "16-mixed", "bf16-mixed"],
        default="16-mixed",
        help="Training precision"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches for gradient accumulation"
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=None,
        help="Number of batches for training"
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=None,
        help="Number of batches for validation"
    )

    # Logging and saving
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="rlhf-reward-model",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=0.25,
        help="How often to run validation (fraction of epoch)"
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="How often to log metrics"
    )
    
    # Additional options
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    return parser.parse_args()

def train_reward_model(args):
    """
    Train a reward model using PyTorch Lightning
    
    Args:
        args: Parsed command line arguments
    """
    # Set seeds for reproducibility
    L.seed_everything(args.seed)
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize logger
    if not args.disable_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=f"reward_model_{timestamp}",
            config=vars(args),
            save_code=True
        )
    else:
        logger = True  # Use default Lightning logger
        
    # Initialize reward model
    print("Initializing reward model...")
    model = RewardModelLM(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.max_epochs,
        use_lora=args.use_lora,
        lora_config={
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM"
        } if args.use_lora else None
    )
        
    # Get data loaders
    print("Loading and processing datasets...")
    train_loader, val_loader, test_loader = get_anthropic_rlhf_data(
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        model_name=args.model_name
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
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
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
    
    # Close wandb run if it was used
    if not args.disable_wandb:
        wandb.finish()
    
    return model, trainer

def main():
    """Main training function with command line argument parsing"""
    args = parse_args()
    
    # Print all arguments for logging purposes
    print("\nTraining with the following parameters:")
    for arg, value in vars(args).items():
        print(f"{arg:.<30} {value}")
    print()
    
    # Train model
    trained_model, trainer = train_reward_model(args)
    
    print("\nTraining completed!")
    
    # Print best validation loss
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    print(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.4f}")

if __name__ == "__main__":
    main()