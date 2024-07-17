#!.venv/bin/python3
"""train script with pytorch-lightning"""

__version__ = "0.2"

import os
import sys
from pathlib import Path
from typing import *

import torch.nn.functional as F
import torch.optim as optim
import timm
from timm.scheduler import CosineLRScheduler
import lightning as L

# const
ROOT_DIR = Path.cwd()
CUSTOM_SCRIPT_DIR = ROOT_DIR / "src"

sys.path.append(str(CUSTOM_SCRIPT_DIR))
from models import ToyModel


class LitToyModel(L.LightningModule):

    def __init__(
        self,
        # model params
        in_channels: int = 3,
        out_channels: int = 32,
        num_class: int = 10,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        # optimizer params
        lr: float = 1e-3,
        # scheduler params
        t_initial: int = 300,
        lr_min: float = 1e-6,
        warmup_t: int = 20,
        warmup_lr_init: float = 1e-6,
        cycle_limit: int = 1,
        cycle_decay: float = 0.5,
        cycle_mul: float = 1.0,
    ) -> None:
        self.num_class = num_class
        super().__init__()
        # save hyperprameters
        self.save_hyperparameters()

        self.model = ToyModel(
            in_channels,
            out_channels,
            num_class,
            kernel_size,
            stride,
            padding,
            dilation,
        )

        self.optim_params = {
            "lr": lr,
        }
        self.scheduler_params = {
            "t_initial": t_initial,
            "lr_min": lr_min,
            "warmup_t": warmup_t,
            "warmup_lr_init": warmup_lr_init,
            "cycle_limit": cycle_limit,
            "cycle_decay": cycle_decay,
            "cycle_mul": cycle_mul,
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        pass

    def configure_optimizers(self):
        # TODO timmのoptimizerに変更
        optimizer = optim.AdamW(self.parameters(), **self.optim_params)
        scheduler = CosineLRScheduler(optimizer=optimizer, **self.scheduler_params)

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, metric) -> None:
        scheduler.step(epoch=self.current_epoch)
