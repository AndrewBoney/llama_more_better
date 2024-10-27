import lightning as L
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

class RewardModelLM(L.LightningModule):
    def __init__(self, model, learning_rate=1e-4, weight_decay=0.01, num_epochs=10):
        super().__init__()
        self.model = model
        self.model.score = nn.Linear(self.model.config.hidden_size, 1)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs

    def forward(self, *args, **kwargs):
        if "output_hidden_states" in kwargs.items():
            raise ValueError("`output_hidden_states` can't be set as hidden state outputs are required")
        output = self.model(*args, **kwargs, output_hidden_states=True)

        logits = output.hidden_states[-1][:, -1, :]

        with torch.amp.autocast(self.device.type):
            return self.model.score(logits)

    def training_step(self, batch, batch_idx):
        chosen_rewards = self(batch["chosen_tokens"])
        rejected_rewards = self(batch["rejected_tokens"])

        # Compute loss (we want chosen_rewards to be higher than rejected_rewards)
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        chosen_rewards = self(batch["chosen_tokens"])
        rejected_rewards = self(batch["rejected_tokens"])

        # Compute validation loss
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        
        # Compute accuracy (percentage where chosen_rewards > rejected_rewards)
        accuracy = (chosen_rewards > rejected_rewards).float().mean()

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        
        return {"val_loss": loss, "val_accuracy": accuracy}

    def test_step(self, batch, batch_idx):
        chosen_rewards = self(batch["chosen_tokens"])
        rejected_rewards = self(batch["rejected_tokens"])

        # Compute test loss
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        
        # Compute accuracy
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        # Compute average reward difference
        reward_diff = (chosen_rewards - rejected_rewards).mean()

        # Log metrics
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        self.log("test_reward_diff", reward_diff)
        
        return {
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_reward_diff": reward_diff
        }

    def configure_optimizers(self):
        # Create optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Create learning rate scheduler
        scheduler = CosineAnnealingLR(
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