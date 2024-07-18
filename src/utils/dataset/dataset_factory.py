#!.venv/bin/python3
"""dataset class for WikiText103"""

# import
from pathlib import Path
from typing import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class WikiText103(Dataset):
    def __init__(
        self,
        token_file: str | Path,
        tokenizer: Any,
    ) -> None:
        super().__init__()

        self.raw_data = Path(token_file).read_text().splitlines(True)
        self.tokenizer = tokenizer

        self.dataset = list(
            tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(
                    line.strip(" ").replace("\n", "[SEP]").replace("<unk>", "[UNK]"),
                    max_length=512,
                    truncation=True,
                    padding="max_length",
                )
            )
            for line in self.raw_data
        )

    def __len__(self) -> None:
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    pass


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from pathlib import Path
    from tqdm import tqdm

    ROOT_DIR = Path.cwd()
    DATA_DIR = ROOT_DIR / "data/WikiText103"

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="bert-base-cased", force_download=True
    )
    train_set = WikiText103(DATA_DIR / "wiki.valid.tokens", tokenizer)
    print(train_set[:10])

    pass
