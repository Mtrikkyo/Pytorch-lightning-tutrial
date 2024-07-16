#!.venv/bin/python3
"""train-script with pytorch-lightning."""

# version
__version__ = "0.0"

# import
import re
import os
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
import lightning as L

from models import LitToyModel

# args
parser = ArgumentParser()

args = parser.parse_args()

# const
ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data"


def main(args: Namespace):
    # lightning model instance
    toy_model = LitToyModel()

    # dataloader
    trian_set = MNIST(DATA_DIR, True)
    trian_loader = DataLoader(trian_set)

    # train
    trainer = L.Trainer()
    trainer.fit(toy_model, train_dataloaders=trian_loader)

    pass


if __name__ == "__main__":
    main(args)
    pass