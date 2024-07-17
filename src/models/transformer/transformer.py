#  #!/usr/bin/python3
"""transformer class"""

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
        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(
                nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
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
