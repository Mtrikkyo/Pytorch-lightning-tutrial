#!.venv/bin/python3
"""train script with pytorch-lightning"""

__version__ = "0.0"

import os
import sys
from pathlib import Path
from typing import *

import torch.nn.functional as F
import torch.optim as optim
import lightning as L

# const
ROOT_DIR = Path.cwd()
CUSTOM_SCRIPT_DIR = ROOT_DIR / "src"

sys.path.append(str(CUSTOM_SCRIPT_DIR))
from models import ToyModel


class LitToyModel(L.LightningModule):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 32,
        num_class: int = 10,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ) -> None:
        self.num_class = num_class
        super().__init__()

        self.model = ToyModel(
            in_channels,
            out_channels,
            num_class,
            kernel_size,
            stride,
            padding,
            dilation,
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        return loss

    def configure_optimizers(self):
        # TODO timmのoptimizerに変更
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)