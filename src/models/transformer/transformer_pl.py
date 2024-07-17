#  #!/usr/bin/python3
"""transformer class using lightning"""
# version
__version__ = "0.1"

# import
import torch
import torch.nn as nn
import lightning as L


class LitTransformer(L.LightningDataModule):

    def __init__(self) -> None:
        super().__init__()

    pass
