#!.venv/bin/python3
"""train-script with pytorch-lightning."""

# import
import re
import os
from pathlib import Path
from argparse import ArgumentParser
import torch
from torch.utils.data import Dataset, DataLoader

# args
parser = ArgumentParser()

args = parser.parse_args()

# const
ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data"


# class WikiText103(Dataset):
#     """"""

#     def __init__(
#         self, root: str = DATA_DIR / "wikitext-103", split: str = "train"
#     ) -> None:
#         super().__init__()
#         self.data_path = root / f"wiki.{split}.tokens"

#         # TODO make class to prepare dataset
#         # read dataset
#         self.row_data = self.data_path.read_text()
#         self.row_data = " \n" + self.row_data

#         # split article
#         self.articles = re.split("( \n \n = [^=]*[^=] = \n \n )", self.row_data)

#         # split article to headding and text.
#         self.headdings = [article[7:-7] for article in self.articles[1::2]]
#         self.texts = [article for article in self.articles[2::2]]

#     def __len__(self):
#         pass

#     def __item__(self):
#         pass


# main script
def main():

    # make instance of DataLoader

    pass


if __name__ == "__main__":
    # main()
    # train_set = WikiText103(split="train")
    # print(train_set.headdings)
    # print(train_set.headdings[111])
    # print(len(train_set.articles))
    pass
