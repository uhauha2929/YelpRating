from typing import Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dan import DeepAveragingNetworks


class HierarchicalJointModelDAN(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 embedding_size: int,
                 hidden_size: int,
                 label_size: int = 9,
                 user_feats_dim: int = 20):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.label_size = label_size

        self.user_feats_dim = user_feats_dim

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.sentence_encoder = DeepAveragingNetworks(embedding_size)
        self.review_encoder = DeepAveragingNetworks(self.sentence_encoder.get_output_dim())
        self.product_encoder = DeepAveragingNetworks(self.review_encoder.get_output_dim() + self.user_feats_dim)

        self.review_feedforward = nn.Sequential(
            nn.Linear(self.sentence_encoder.get_output_dim() + user_feats_dim, 1),
            nn.SELU()
        )

        self.product_feedforward = nn.Sequential(
            nn.Linear(self.product_encoder.get_output_dim(), self.label_size),
            nn.SELU()
        )

        if self.user_feats_dim > 0:
            self.user_feats_weights = nn.Parameter(torch.ones(self.user_feats_dim), requires_grad=True)

    def load_word_embedding(self, embedding_file: str):
        embedding = torch.FloatTensor(np.load(embedding_file))
        assert self.embedding.weight.shape == embedding.size(), \
            'Shape of weight does not match num_embeddings and embedding_dim'
        self.embedding.weight.data.copy_(embedding)

    def forward(self,
                inputs: torch.Tensor,
                user_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [B, 10, 20, 30]
        p_list = []
        for i, p in enumerate(inputs):
            # [10, 20, 30]
            r_mask = p.sum(-1) != 0
            r_list = []
            for j, r in enumerate(p):
                # [20, 30]
                s_mask = r != 0
                r = word_dropout(r, training=self.training)
                s_embedded = self.embedding(r)
                # [20, 30, E]
                r_hn = self.sentence_encoder(s_embedded, s_mask)
                # [20, H] -> [1, 20, H]
                r_list.append(r_hn.unsqueeze(0))

            r_batch = torch.cat(r_list, dim=0)
            # [10, 20, H]
            p_hn = self.review_encoder(r_batch, r_mask)
            # [10, H] -> [1, 10, H]
            p_list.append(p_hn.unsqueeze(0))

        p_batch = torch.cat(p_list, dim=0)
        # [B, 10, H]
        if self.user_feats_dim > 0:
            p_batch = torch.cat([p_batch, F.normalize(user_feats) * self.user_feats_weights], -1)
        # [B, 10, H+F]
        r_stars = self.review_feedforward(p_batch).squeeze()
        # [B, 10]
        h_n = self.product_encoder(p_batch)
        # [B, H]
        p_stars = self.product_feedforward(h_n.squeeze())
        # [B]
        return p_stars, r_stars
