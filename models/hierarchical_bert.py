# -*- coding: utf-8 -*-
# @Time    : 2019/12/25 11:14
# @Author  : uhauha2929
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, CnnEncoder


class HierarchicalJointModelBERT(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 bert_dim: int = 768,
                 label_size: int = 9,
                 dropout: float = 0.2,
                 user_feats_dim: int = 20):
        super().__init__()

        self.hidden_size = hidden_size
        self.bert_dim = bert_dim

        self.label_size = label_size
        self.user_feats_dim = user_feats_dim
        self.dropout = nn.Dropout(dropout)

        # self.sentence_rnn = PytorchSeq2VecWrapper(nn.GRU(bert_dim,
        #                                                  hidden_size,
        #                                                  batch_first=True,
        #                                                  bidirectional=True))
        self.projection = torch.nn.Linear(bert_dim, self.hidden_size)
        self.sentence_cnn = CnnEncoder(self.hidden_size, num_filters=16)

        self.review_rnn = PytorchSeq2VecWrapper(nn.GRU(self.sentence_cnn.get_output_dim(),
                                                       hidden_size,
                                                       batch_first=True,
                                                       bidirectional=True))

        self.product_rnn = nn.GRU(hidden_size * 2 + self.user_feats_dim,
                                  hidden_size,
                                  batch_first=True,
                                  bidirectional=True)

        self.review_feedforward = nn.Sequential(
            nn.Linear(hidden_size * 2 + self.user_feats_dim, hidden_size // 2),
            nn.ELU(),
            self.dropout,
            nn.Linear(hidden_size // 2, 1)
        )

        self.product_feedforward = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ELU(),
            self.dropout,
            nn.Linear(hidden_size // 2, self.label_size)
        )

        if self.user_feats_dim > 0:
            self.user_feats_weights = nn.Parameter(torch.ones(self.user_feats_dim))

    def forward(self,
                inputs: torch.Tensor,
                user_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        inputs = self.dropout(self.projection(inputs))
        # [B, 10, 20, 30, E]
        p_list = []
        for i, p in enumerate(inputs):
            # [10, 20, 30, E]
            r_mask = p.sum(-1).sum(-1) != 0
            r_list = []
            for j, r in enumerate(p):
                # [20, 30, E]
                s_mask = r.sum(-1) != 0
                r_hn = self.sentence_cnn(r, s_mask)
                # [20, H] -> [1, 20, H]
                r_list.append(r_hn.unsqueeze(0))

            r_batch = torch.cat(r_list, dim=0)
            # [10, 20, H]
            p_hn = self.review_rnn(r_batch, r_mask)
            # [10, H] -> [1, 10, H]
            p_list.append(p_hn.unsqueeze(0))

        p_batch = torch.cat(p_list, dim=0)
        # [B, 10, H]
        if self.user_feats_dim > 0:
            p_batch = torch.cat([p_batch, F.normalize(user_feats) * self.user_feats_weights], -1)
        # [B, 10, H+F]
        r_stars = self.review_feedforward(p_batch).squeeze()
        # [B, 10]
        _, h_n = self.product_rnn(p_batch)
        h_n = torch.cat([h_n[:1, :, :], h_n[1:, :, :]], -1)
        # [B, H]
        p_stars = self.product_feedforward(h_n.squeeze())
        # [B]
        return p_stars, r_stars
