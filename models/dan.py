# -*- coding: utf-8 -*-
# @Time    : 2020/1/7 11:09
# @Author  : uhauha2929
import torch
import torch.nn as nn


def word_dropout(inputs: torch.Tensor,
                 padding_id: int = 0,
                 mask: torch.Tensor = None,
                 dropout: float = 0.3,
                 training: bool = True):
    # [b, s]
    if not training:
        return inputs
    prob = torch.rand(inputs.size())
    if mask is not None:
        prob[mask == padding_id] = 1
    inputs = inputs.clone()  # fix inplace error
    inputs[prob < dropout] = padding_id
    return inputs


class DeepAveragingNetworks(nn.Module):

    def __init__(self, embedding_dim: int, n_layers: int = 3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.feedforward = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim)
                                          for _ in range(n_layers)])
        self.activation = nn.SELU()

    def get_output_dim(self):
        return self.embedding_dim

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # inputs: [b, s, h], mask: [b, s]
        if mask is None:
            inputs = inputs.mean(-2)
        else:
            inputs = torch.mean(inputs * (mask.float().unsqueeze(-1)), -2)
        for layer in self.feedforward:
            inputs = self.activation(layer(inputs))
        return inputs


class DANWrapper(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_layers: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.dan = DeepAveragingNetworks(embedding_dim, n_layers)

    def forward(self, inputs: torch.Tensor, *args) -> torch.Tensor:
        mask = (inputs != 0).float()
        inputs = self.embed(inputs)
        return self.dan(inputs, mask)

    def load_embed_matrix(self, matrix: torch.Tensor):
        self.embed.weight.data.copy_(matrix)
