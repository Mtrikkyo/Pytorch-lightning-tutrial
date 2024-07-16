#!.venv/bin/python3
"""train script with pytorch-lightning"""
from typing import *
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from .toy_model import ToyModel


class LitToyModel(L.LightningModule):

    def __init__(self, num_class) -> None:
        self.num_class = num_class
        super().__init__()

        self.model = ToyModel()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        return loss

    def configure_optimizers(self):
        # TODO timmのoptimizerに変更
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
