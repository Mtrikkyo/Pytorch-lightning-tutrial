#!.venv/bin/python3
"""train-script(Transformer) with pytorch-lightning."""

# import
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from transformers import AutoTokenizer


from models.transformer import LitTransformer
from utils.dataset.dataset_factory import WikiText103

# args
parser = ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=["gpt2", "hopfield"],
    help="""""",
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["wikitext103"],
    help="""""",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    default="bert-base-cased",
    help="""pretrained model name or path to get tonizer from`transformers` library. """,
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="""batch-size of data-loaders.""",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="data",
    help="""""",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="result",
    help="""directory to save wandblog and checkpoint.""",
)
parser.add_argument(
    "--project_name",
    type=str,
    default="HSDT-WikiText103",
    help="""project name on Werights and Biasis.""",
)
parser.add_argument(
    "--run_name",
    type=str,
    default="HSDT-WikiText103",
    help="""project name on Werights and Biasis.""",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    default=None,
    help="""checkpoint path to resume training""",
)
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
SAVE_DIR = Path(args.data_dir)


def main(args: Namespace):

    # model instance
    if args.model == "gpt2":
        model = LitTransformer()
        pass

    elif args.model == "hopfield":
        pass
    else:
        # error
        pass

    # dataset instance
    if args.dataset == "wikitext103":
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
        )
        train_laoder = DataLoader(
            dataset=WikiText103(
                DATA_DIR / "WikiText103/wiki.train.tokens",
                tokenizer=tokenizer,
            ),
            batch_size=args.batch_size,
        )
        val_laoder = DataLoader(
            dataset=WikiText103(
                DATA_DIR / "WikiText103/wiki.valid.tokens",
                tokenizer=tokenizer,
            ),
            batch_size=args.batch_size,
        )
        pass

    pass
    # logger
    wandb_logger = WandbLogger(
        project=args.project_name,
        name=args.run_name,
        save_dir=args.save_dir,
        log_model=True,
    )
    csv_logger = CSVLogger(
        SAVE_DIR / args.project_name / args.run_name,
    )

    # train
    trainer = L.Trainer(
        max_epochs=args.epoch,
        logger=[wandb_logger, csv_logger],
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_laoder,
        val_dataloaders=val_laoder,
        ckpt_path=args.ckpt_path,
    )


if __name__ == "__main__":
    pass
