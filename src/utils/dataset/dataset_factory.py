# #!/usr/bin/python3
"""dataset class for WikiText103"""

# import
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class WikiText103(Dataset):
    pass


if __name__ == "__main__":
    from transformers import BertTokenizer
    from pathlib import Path

    ROOT_DIR = Path.cwd().parents[2]
    DATA_DIR = ROOT_DIR / "data/WikiText103"

    # train_data = (DATA_DIR / "wiki.train.tokens").read_text()
    valid_data = (DATA_DIR / "wiki.valid.tokens").read_text()
    print(valid_data)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    pass
