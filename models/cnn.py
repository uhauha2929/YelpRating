# -*- coding: utf-8 -*-
# @Time    : 2019/5/5 20:28
# @Author  : uhauha2929
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders import CnnEncoder


class SimpleCNN(nn.Module):

    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, emb_dim)

        self.cnn_encoder = CnnEncoder(emb_dim, 128, output_dim=hid_dim)

        self.fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hid_dim // 2, 9)
        )

    def load_embed_matrix(self, matrix: torch.Tensor):
        self.embed.weight.data.copy_(matrix)

    def forward(self, inputs, *args):
        batch_size = inputs.size(0)
        inputs = inputs.view(batch_size, -1)
        inputs = self.embed(inputs)
        outputs = self.cnn_encoder(inputs)
        outputs = self.fc(outputs)
        return outputs
