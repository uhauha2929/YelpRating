# -*- coding: utf-8 -*-
# @Time    : 2019/12/25 11:14
# @Author  : uhauha2929
from typing import Tuple
import transformers as pt
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

from config import Config


class HierarchicalJointModelBERT(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        for k, v in vars(config).items():
            setattr(self, k, v)

        self.bert = pt.BertModel.from_pretrained(self.pretrained_data_dir)

        self.embedding_dim = self.bert.config.to_dict()['hidden_size']

        self.dropout = nn.Dropout(self.dropout)

        self.sentence_rnn = PytorchSeq2VecWrapper(nn.GRU(self.embedding_dim,
                                                         self.hidden_dim,
                                                         batch_first=True,
                                                         bidirectional=True))

        self.review_rnn = PytorchSeq2VecWrapper(nn.GRU(self.sentence_rnn.get_output_dim(),
                                                       self.hidden_dim,
                                                       batch_first=True,
                                                       bidirectional=True))

        self.product_rnn = nn.GRU(self.hidden_dim * 2 + self.user_feats_dim,
                                  self.hidden_dim,
                                  batch_first=True,
                                  bidirectional=True)

        self.review_feedforward = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.user_feats_dim,
                      self.hidden_dim // 2),
            self.dropout,
            nn.ELU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )

        self.product_feedforward = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim // 2),
            self.dropout,
            nn.ELU(),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )

        if self.user_feats_dim > 0:
            self.user_feats_weights = nn.Parameter(torch.ones(self.user_feats_dim))

    def forward(self,
                inputs: torch.Tensor,
                user_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # [B, 10, 20, 30]
        p_list = []
        for i, p in enumerate(inputs):
            # [10, 20, 30]
            r_mask = (p.sum(-1) != 0).float()
            r_list = []
            for j, r in enumerate(p):
                # [20, 30]
                s_mask = (r != 0).float()
                embedded = self.bert(r, attention_mask=s_mask)[0]
                # [20, 30, E]
                r_hn = self.sentence_rnn(embedded, s_mask)
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
            p_batch = torch.cat([p_batch, user_feats * self.user_feats_weights], -1)
        # [B, 10, H+F]
        r_stars = self.review_feedforward(p_batch).squeeze()
        # [B, 10]
        _, h_n = self.product_rnn(p_batch)
        h_n = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], -1)
        # [B, H]
        p_stars = self.product_feedforward(h_n.squeeze())
        # [B]
        return p_stars, r_stars
