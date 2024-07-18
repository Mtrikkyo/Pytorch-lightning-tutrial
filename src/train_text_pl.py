#!.venv/bin/python3
"""train-script(Transformer) with pytorch-lightning."""

# import
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelSummary
from transformers import AutoTokenizer


from models.transformer import LitTransformer
from utils.dataset.dataset_factory import WikiText103

# args
parser = ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=["gpt2", "hopfield"],
    default="gpt2",
    help="""""",
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["wikitext103"],
    default="wikitext103",
    help="""""",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    default="bert-base-cased",
    help="""pretrained model name or path to get tonizer from`transformers` library. """,
)
parser.add_argument(
    "--epoch",
    type=int,
    default=10,
)
parser.add_argument_group("Scheduler params")
parser.add_argument(
    "--lr",
    type=int,
    default=6.25e-5,
)
parser.add_argument(
    "--t_initial",
    type=int,
    default=10,
)
parser.add_argument(
    "--lr_min",
    type=float,
    default=1e-6,
)
parser.add_argument(
    "--warmup_t",
    type=int,
    default=20,
)
parser.add_argument(
    "--warmup_lr_init",
    type=float,
    default=1e-6,
)
parser.add_argument(
    "--cycle_limit",
    type=int,
    default=1,
)
parser.add_argument(
    "--cycle_decay",
    type=float,
    default=0.5,
)
parser.add_argument(
    "--cycle_mul",
    type=float,
    default=1.0,
)
parser.add_argument_group("Model Parameters")
parser.add_argument(
    "--embed_dim",
    type=int,
    default=512,
)
parser.add_argument(
    "--hidden_dim",
    type=int,
    default=512,
)
parser.add_argument(
    "--num_heads",
    type=int,
    default=8,
)
parser.add_argument(
    "--num_layers",
    type=int,
    default=8,
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.0,
)
parser.add_argument(
    "--initializer_range",
    type=float,
    default=0.0,
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=5,
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

    # tokenizer instance
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
    )

    # model instance
    if args.model == "gpt2":
        model = LitTransformer(args, vocab_size=tokenizer.vocab_size)
        pass

    elif args.model == "hopfield":
        pass
    else:
        print("unko")
        # error
        pass

    # dataset instance
    if args.dataset == "wikitext103":

        # train_loader = DataLoader(
        #     dataset=WikiText103(
        #         DATA_DIR / "WikiText103/wiki.train.tokens",
        #         tokenizer=tokenizer,
        #     ),
        #     batch_size=args.batch_size,
        # )
        val_loader = DataLoader(
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
        train_dataloaders=val_loader,
        # val_dataloaders=val_loader,
        ckpt_path=args.ckpt_path,
    )


if __name__ == "__main__":
    main(args)
    pass
