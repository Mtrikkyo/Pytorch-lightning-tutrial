#!.venv/bin/python3
"""train-script with pytorch-lightning."""

# version
__version__ = "0.1"

# import
import re
import os
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
import lightning as L

from models import LitToyModel

# args
parser = ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    choices=["mnist", "cifar10", "cifar100"],
    default="mnist",
    help=""" """,
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="data",
    help=""" """,
)
parser.add_argument(
    "--epoch",
    type=int,
    default=300,
    help=""" """,
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help=""" """,
)

args = parser.parse_args()

# const
ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / args.data_dir


def main(args: Namespace):

    # dataloader
    match args.dataset:
        case "mnist":

            # lightning model instance
            model = LitToyModel(in_channels=1, num_class=10)

            trian_set = MNIST(DATA_DIR, True, transform=v2.ToTensor(), download=True)
            trian_loader = DataLoader(trian_set, batch_size=args.batch_size)
            val_set = MNIST(DATA_DIR, False, transform=v2.ToTensor(), download=True)
            val_loader = DataLoader(val_set, batch_size=args.batch_size)

        case "cifar10":

            # lightning model instance
            model = LitToyModel(in_channels=3, num_class=10)

            trian_set = CIFAR10(DATA_DIR, True, transform=v2.ToTensor(), download=True)
            trian_loader = DataLoader(trian_set, batch_size=args.batch_size)
            val_set = CIFAR10(DATA_DIR, False, transform=v2.ToTensor(), download=True)
            val_loader = DataLoader(val_set, batch_size=args.batch_size)

        case "cifar100":

            # lightning model instance
            model = LitToyModel(in_channels=3, num_class=100)

            trian_set = CIFAR100(DATA_DIR, True, transform=v2.ToTensor(), download=True)
            trian_loader = DataLoader(trian_set, batch_size=args.batch_size)
            val_set = CIFAR100(DATA_DIR, False, transform=v2.ToTensor(), download=True)
            val_loader = DataLoader(val_set, batch_size=args.batch_size)
    # train
    trainer = L.Trainer(max_epochs=args.epoch)
    trainer.fit(model, train_dataloaders=trian_loader, val_dataloaders=val_loader)

    pass


if __name__ == "__main__":
    main(args)
    pass
