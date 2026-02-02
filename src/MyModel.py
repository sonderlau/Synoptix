from typing import Any

import torch
from lightning.pytorch import LightningModule
from torch import Tensor

from utils.base import PredictionType


class MyModel(LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):

        # model forward pass
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss)

    def predict_step(self, *args: Any, **kwargs: Any) -> PredictionType:

        result = PredictionType()

        result.pred = torch.tensor()
        result.target = torch.tensor()

        return result

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1.5e-4)