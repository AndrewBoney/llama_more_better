import lightning as L

import torch

from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional, Dict, Any

class RewardModelLM(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        use_lora: bool = False,
        lora_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Reward Model with optional LoRA support
        
        Args:
            model_name: HuggingFace model identifier
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            num_epochs: Number of training epochs
            use_lora: Whether to use LoRA
            lora_config: LoRA configuration parameters. If None and use_lora=True, uses defaults
        """
        super().__init__()
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add reward head
        self.model.score = nn.Linear(self.model.config.hidden_size, 1)
        
        # Apply LoRA if specified
        if use_lora:
            default_lora_config = {
                "r": 8,  # LoRA attention dimension
                "lora_alpha": 16,  # Alpha scaling
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # Which modules to apply LoRA to
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": TaskType.CAUSAL_LM
            }
            
            # Use provided config or defaults
            lora_config = lora_config or default_lora_config
            
            peft_config = LoraConfig(
                **lora_config
            )
            
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        # Save hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs

    def forward(self, *args, **kwargs):
        if "output_hidden_states" in kwargs:
            raise ValueError("`output_hidden_states` can't be set as hidden state outputs are required")
        
        output = self.model(*args, **kwargs, output_hidden_states=True)
        logits = output.hidden_states[-1][:, -1, :]
        
        with torch.amp.autocast(self.device.type):
            return self.model.score(logits)

    def get_rewards(self, batch):
        chosen_rewards = self(**batch["chosen_tokens"])
        rejected_rewards = self(**batch["rejected_tokens"])
        return chosen_rewards, rejected_rewards

    def training_step(self, batch, batch_idx):
        chosen_rewards, rejected_rewards = self.get_rewards(batch)
        
        # Compute loss (we want chosen_rewards to be higher than rejected_rewards)
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        chosen_rewards, rejected_rewards = self.get_rewards(batch)
        
        # Compute validation loss
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        
        # Compute accuracy (percentage where chosen_rewards > rejected_rewards)
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        
        return {"val_loss": loss, "val_accuracy": accuracy}

    def test_step(self, batch, batch_idx):
        chosen_rewards, rejected_rewards = self.get_rewards(batch)
        
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        reward_diff = (chosen_rewards - rejected_rewards).mean()
        
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        self.log("test_reward_diff", reward_diff)
        
        return {
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_reward_diff": reward_diff
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs,
            eta_min=self.learning_rate/100
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }