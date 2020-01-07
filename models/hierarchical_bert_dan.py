from typing import Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dan import DeepAveragingNetworks


class HierarchicalJointModelBERTDAN(nn.Module):

    def __init__(self,
                 bert_dim: int,
                 hidden_size: int,
                 label_size: int = 9,
                 user_feats_dim: int = 20):
        super().__init__()

        self.bert_dim = bert_dim
        self.hidden_size = hidden_size
        self.label_size = label_size

        self.user_feats_dim = user_feats_dim

        self.projection = nn.Linear(bert_dim, hidden_size)

        self.sentence_encoder = DeepAveragingNetworks(hidden_size)
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

    def forward(self,
                inputs: torch.Tensor,
                user_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [B, 10, 20, 30, E]
        inputs = self.projection(inputs)
        p_list = []
        for i, p in enumerate(inputs):
            # [10, 20, 30, E]
            r_mask = p.sum(-1).sum(-1) != 0
            r_list = []
            for j, r in enumerate(p):
                # [20, 30, E]
                s_mask = r.sum(-1) != 0
                # [20, 30, E]
                r_hn = self.sentence_encoder(r, s_mask)
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
