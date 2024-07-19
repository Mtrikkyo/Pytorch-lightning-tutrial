#!.venv/bin/python3
"""transformer class"""
from argparse import Namespace

import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.tokens_embeddings = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim,
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim,
        )
        self.dropout = nn.Dropout(dropout)

        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(
                nn.MultiheadAttention(
                    embed_dim, num_heads, dropout=dropout, batch_first=True
                )
            )
            self.feed_forwards.append(
                nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embed_dim),
                )
            )
            self.layer_norms_1.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.layer_norms_2.append(nn.LayerNorm(embed_dim, eps=1e-12))

    def forward(self, x):
        _, seq_length = x.shape

        # token_embedding
        h = self.tokens_embeddings(x)

        # position_embedding
        positions = torch.arange(seq_length, device=x.device)
        pos = self.position_embeddings(positions).expand_as(h)
        # print(pos.shape)
        h = h + pos

        h = self.dropout(h)

        # attn_mask
        attn_mask = torch.full((seq_length, seq_length), float("-inf"), device=h.device)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        if str(h.device).startswith("mps"):
            attn_mask = torch.nan_to_num(attn_mask, nan=0.0)

        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(
            self.layer_norms_1, self.attentions, self.layer_norms_2, self.feed_forwards
        ):
            h = layer_norm_1(h)

            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
        return h


class TransformerWithLMHead(nn.Module):
    def __init__(self, args: Namespace, vocab_size: int):
        super().__init__()
        self.args = args
        self.transformer = Transformer(
            num_embeddings=vocab_size,
            embed_dim=self.args.embed_dim,
            hidden_dim=self.args.hidden_dim,
            num_heads=self.args.num_heads,
            num_layers=self.args.num_layers,
            dropout=self.args.dropout,
        )
        self.lm_head = nn.Linear(self.args.embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.transformer.tokens_embeddings.weight  # Tie weights
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        x = self.transformer(x)
        x = self.lm_head(x)

        return x
