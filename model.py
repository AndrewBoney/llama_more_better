import lightning as L

import torch

from torch import nn

class RewardModelLM(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.score = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, *args, **kwargs):
        if "output_hidden_states" in kwargs.items():
            raise ValueError("`output_hidden_states` can't be set as hidden state outputs are required")
        output = self.model(*args, **kwargs, output_hidden_states = True)

        logits = output.hidden_states[-1][:, -1, :]

        with torch.amp.autocast(self.device.type):
            return self.model.score(logits)