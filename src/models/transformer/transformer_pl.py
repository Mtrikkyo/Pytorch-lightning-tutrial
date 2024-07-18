#!python3
"""transformer class using lightning"""
# version
__version__ = "0.1"

# import
import sys
from pathlib import Path

import torch
import torch.nn as nn
import lightning as L

# const
ROOT_DIR = Path.cwd()
CUSTOM_SCRIPT_DIR = ROOT_DIR / "src"

sys.path.append(str(CUSTOM_SCRIPT_DIR))
from models.transformer import TransformerWithLMHead


class LitTransformer(L.LightningDataModule):

    def __init__(self) -> None:
        super().__init__()

        self.model = TransformerWithLMHead()

    def training_step(self, batch, batch_idx):
        x = batch
        pass
