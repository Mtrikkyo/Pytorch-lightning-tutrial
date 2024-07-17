#!.venv/bin/python3
"""train script with pytorch-lightning"""

__version__ = "0.2"

import os
import sys
from pathlib import Path
from typing import *

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from timm.utils.metrics import accuracy
from timm.scheduler import CosineLRScheduler
from timm.data import Mixup
import lightning as L

# const
ROOT_DIR = Path.cwd()
CUSTOM_SCRIPT_DIR = ROOT_DIR / "src"

sys.path.append(str(CUSTOM_SCRIPT_DIR))
from models.toy import ToyModel


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
        # mixup params
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        cutmix_minmax: Optional[float] = None,
        mixup_prob: float = 1.0,
        switch_prob: float = 0.5,
        mixup_mode=None,
        label_smoothing: float = 0.1,
        # loss
        valid_loss_fn=nn.CrossEntropyLoss(),
    ) -> None:
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
        self.mixup_params = {
            "mixup_alpha": mixup_alpha,
            "cutmix_alpha": cutmix_alpha,
            "cutmix_minmax": cutmix_minmax,
            "prob": mixup_prob,
            "switch_prob": switch_prob,
            "mode": mixup_mode,
            "label_smoothing": label_smoothing,
            "num_classes": num_class,
        }

        self.mixup_fn = None
        if (
            self.mixup_params["mixup_alpha"] > 0
            or self.mixup_params["cutmix_alpha"] > 0
            or self.mixup_params["cutmix_minmax"] is not None
        ):
            self.mixup_fn = Mixup(**self.mixup_params)

        self.valid_loss_fn = valid_loss_fn

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_copy = y
        if self.mixup_fn is not None:
            x, y = self.mixup_fn(x, y)
        y_hat = self.model(x)

        top1, top5 = accuracy(y_hat, y_copy, topk=(1, 5))
        loss = self.valid_loss_fn(y_hat, y)

        self.log("train/top1", top1.item(), on_step=False, on_epoch=True)
        self.log("train/loss", loss.item(), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        x, y = batch
        y_hat = self.model(x)

        top1, top5 = accuracy(y_hat, y, topk=(1, 5))
        loss = self.valid_loss_fn(y_hat, y)

        self.log("valid/top1", top1.item(), on_step=False, on_epoch=True)
        self.log("valid/loss", loss.item(), on_step=False, on_epoch=True)

        return

    def configure_optimizers(self):
        # TODO timmのoptimizerに変更
        optimizer = optim.AdamW(self.parameters(), **self.optim_params)
        scheduler = CosineLRScheduler(optimizer=optimizer, **self.scheduler_params)

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, metric) -> None:
        scheduler.step(epoch=self.current_epoch)
