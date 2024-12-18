import lightning as L

import torch

from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional, Dict, Any

class MLP(nn.Module):
    def __init__(self, hidden_size : int, dropout_rate : float, bias : bool):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2, bias = bias),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1, bias = bias)
        ])

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x

class RewardModelLM(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        use_lora: bool = False,
        lora_config: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        dropout : float = 0.4
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
            bias: Whether to use bias in classification head
            dropout: Dropout rate for classification head
        """
        super().__init__()
        self.save_hyperparameters()

        # Load base model
        # TODO: 
        #   does it makes sense to use AutoModelForSequenceClassification or AutoModelForCausalLM here?
        #   The differences seem to be that AutoModelForSequenceClassification hard replaces the token prediction head with a classifier head 
        #   (with 2 outputs... not sure why 2 and not one if binary?)... and therefore .generate doesn't work AutoModelForSequenceClassification
        #   I think given that the weights of this model will be used later for both generation and reward classification it makes sense to use
        #    AutoModelForCausalLM and manually add an additional classifier head like below.
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
                
        # Apply LoRA if specified
        if use_lora:
            default_lora_config = {
                "r": 8,  # LoRA attention dimension
                "lora_alpha": 16,  # Alpha scaling
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # Which modules to apply LoRA to
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": TaskType.SEQ_CLS #TaskType.CAUSAL_LM
            }
            
            # Use provided config or defaults
            lora_config = lora_config or default_lora_config
            
            peft_config = LoraConfig(
                **lora_config
            )
            
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        # add classification head        
        #self.model.score = nn.Linear(self.model.config.hidden_size, 1, bias = bias)
        self.model.score = MLP(self.model.config.hidden_size, dropout, bias=bias)

        # Save hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.bias = bias

    def forward(self, input_ids, attention_mask=None, *args, **kwargs):
        if "output_hidden_states" in kwargs:
            raise ValueError("`output_hidden_states` can't be set as hidden state outputs are required")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # TODO: 
        #   is there an easy way of doing this without running full model? (i.e. skipping the pass predicting token logits)
        #   no biggie just would save a few wasted FLOPS
        outputs = self.model(input_ids, attention_mask, *args, **kwargs, output_hidden_states=True)

        last_hidden = outputs.hidden_states[-1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        
        # get last actual i.e. non hidden token
        relevant_hidden_states = last_hidden[
            torch.arange(last_hidden.shape[0], device = last_hidden.device), 
            sequence_lengths, 
            :
        ]

        return self.model.score(relevant_hidden_states)

    def get_rewards(self, batch):
        chosen_rewards = self(**batch["chosen"])
        rejected_rewards = self(**batch["rejected"])
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
        
        # Compute accuracy (percentage where chosen_rewards > rejected_rewards
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
            #self.parameters(),
            [p for p in self.model.parameters() if p.requires_grad],
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