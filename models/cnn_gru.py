# -*- coding: utf-8 -*-
# @Time    : 2019/4/19 19:39
# @Author  : uhauha2929
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

from models.encoders import CnnEncoder


class MultiHierarchicalCnnBiGRU(nn.Module):
    name = 'h-cnn-bigru-multi'

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hid_dim: int,
                 num_filters: int = 64,
                 ngram_filter_sizes: Tuple[int, ...] = (2, 3, 4),
                 regress: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.s_enc = CnnEncoder(emb_dim, num_filters, ngram_filter_sizes, hid_dim)
        self.s_rnn = nn.GRU(hid_dim, hid_dim, batch_first=True, bidirectional=True)
        self.r_rnn = nn.GRU(hid_dim, hid_dim, batch_first=True, bidirectional=True)
        self.p_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.SELU(),
            nn.Dropout(),
            nn.Linear(hid_dim // 2, 1 if regress else 9)
        )
        self.r_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.SELU(),
            nn.Dropout(),
            nn.Linear(hid_dim // 2, 1)
        )

    def forward(self, inputs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = F.dropout(self.embedding(inputs))
        # [B, 10, 20, 30, E]
        p_list = []
        for i, p in enumerate(inputs):
            # [10, 20, 30, E]
            r_list = []
            for j, r in enumerate(p):
                # [20, 30, E]
                s_vec = self.s_enc(r)
                r_list.append(s_vec.unsqueeze(0))
            r_batch = torch.cat(r_list, dim=0)
            _, h_n = self.s_rnn(r_batch)
            h_n = h_n[:1, :, :] + h_n[1:, :, :]
            # [1, 10, H]
            p_list.append(h_n)
        p_batch = torch.cat(p_list, dim=0)
        # [B, 10, H]
        r_stars = self.r_fc(p_batch)
        _, h_n = self.r_rnn(p_batch)
        h_n = h_n[:1, :, :] + h_n[1:, :, :]
        # [1, B, H]
        b_stars = self.p_fc(h_n.squeeze())
        # [B, 1]
        return b_stars, r_stars


class MultiUserHierarchicalCnnBiGRU(nn.Module):
    name = 'h-cnn-bigru-multi-user'

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hid_dim: int,
                 num_filters: int = 64,
                 ngram_filter_sizes: Tuple[int, ...] = (2, 3, 4),
                 regress: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.s_enc = CnnEncoder(emb_dim, num_filters, ngram_filter_sizes, hid_dim)
        self.s_rnn = nn.GRU(hid_dim, hid_dim, batch_first=True, bidirectional=True)
        self.r_rnn = nn.GRU(hid_dim + 20, hid_dim, batch_first=True, bidirectional=True)
        self.p_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.SELU(),
            nn.Dropout(),
            nn.Linear(hid_dim // 2, 1 if regress else 9)
        )
        self.r_fc = nn.Sequential(
            nn.Linear(hid_dim + 20, hid_dim // 2),
            nn.SELU(),
            nn.Dropout(),
            nn.Linear(hid_dim // 2, 1)
        )

    def forward(self, inputs: torch.Tensor, user_feats: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = F.dropout(self.embedding(inputs))
        # [B, 10, 20, 30, E]
        p_list = []
        for i, p in enumerate(inputs):
            # [10, 20, 30, E]
            r_list = []
            for j, r in enumerate(p):
                # [20, 30, E]
                s_vec = self.s_enc(r)
                r_list.append(s_vec.unsqueeze(0))
            r_batch = torch.cat(r_list, dim=0)
            _, h_n = self.s_rnn(r_batch)
            h_n = h_n[:1, :, :] + h_n[1:, :, :]
            # [1, 10, H]
            p_list.append(h_n)
        p_batch = torch.cat(p_list, dim=0)
        # [B, 10, H]
        p_batch = torch.cat([p_batch, F.normalize(user_feats)], -1)
        # [B, 10, H+F]
        r_stars = self.r_fc(p_batch)
        _, h_n = self.r_rnn(p_batch)
        h_n = h_n[:1, :, :] + h_n[1:, :, :]
        # [1, B, H]
        b_stars = self.p_fc(h_n.squeeze())
        # [B, 1]
        return b_stars, r_stars
