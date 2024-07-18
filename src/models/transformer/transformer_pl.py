#!python3
"""transformer class using lightning"""
# version
__version__ = "0.1"

# import
from argparse import Namespace
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
import lightning as L

# const
ROOT_DIR = Path.cwd()
CUSTOM_SCRIPT_DIR = ROOT_DIR / "src"

sys.path.append(str(CUSTOM_SCRIPT_DIR))
from models.transformer import TransformerWithLMHead


class LitTransformer(L.LightningModule):

    def __init__(self, args: Namespace, vocab_size: int) -> None:
        super().__init__()

        self.optim_params = {
            "lr": args.lr,
        }
        self.scheduler_params = {
            "t_initial": args.t_initial,
            "lr_min": args.lr_min,
            "warmup_t": args.warmup_t,
            "warmup_lr_init": args.warmup_lr_init,
            "cycle_limit": args.cycle_limit,
            "cycle_decay": args.cycle_decay,
            "cycle_mul": args.cycle_mul,
        }

        self.model = TransformerWithLMHead(args, vocab_size)
        self.train_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def training_step(self, batch, batch_idx):
        x = batch[:-1]
        y = batch[1:]
        y_hat = self.model(x)

        loss = self.train_loss_fn(
            y_hat.view(-1, y_hat.size(-1)),
            y.view(-1),
        )
        perplexity = torch.exp(loss).item()
        self.log("train/loss", loss.item())
        self.log("train/perplexity", perplexity)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[:-1]
        y = batch[1:]
        y_hat = self.model(x)

        loss = self.train_loss_fn(y_hat, y)
        perplexity = torch.exp(loss.item())
        self.log("train/loss", loss.item())
        self.log("train/perplexity", perplexity)

        return

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), **self.optim_params)
        scheduler = CosineLRScheduler(optimizer, **self.scheduler_params)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, metric) -> None:
        scheduler.step(epoch=self.current_epoch)
