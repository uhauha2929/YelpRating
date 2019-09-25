from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.wrappers import GruWrapper


class Multi3GruUser(nn.Module):
    name = 'multi-h-bigru-user'

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hid_dim: int,
                 dropout=0.5,
                 add_user=True):
        super().__init__()
        self._add_user = add_user
        self._embed = nn.Embedding(vocab_size, emb_dim)
        self._w_rnn = GruWrapper(nn.GRU(emb_dim, hid_dim,
                                        batch_first=True,
                                        bidirectional=True),
                                 return_last=True)
        self._s_rnn = GruWrapper(nn.GRU(hid_dim, hid_dim,
                                        batch_first=True,
                                        bidirectional=True),
                                 return_last=True)
        self._r_rnn = GruWrapper(nn.GRU(hid_dim + 20 if add_user else hid_dim, hid_dim,
                                        batch_first=True,
                                        bidirectional=True),
                                 return_last=True)
        self._dropout = nn.Dropout(p=dropout)

        self._r_fc = nn.Sequential(
            nn.Linear(hid_dim + 20 if add_user else hid_dim, hid_dim // 2),
            nn.SELU(),
            self._dropout,
            nn.Linear(hid_dim // 2, 1)
        )

        self._p_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.SELU(),
            self._dropout,
            nn.Linear(hid_dim // 2, 9)
        )

        self.user_feats_weights = nn.Parameter(torch.ones(20))

    def load_embed_matrix(self, matrix: torch.Tensor):
        self._embed.weight.data.copy_(matrix)

    def forward(self,
                inputs: torch.Tensor,
                sent_lengths: torch.Tensor,
                sent_counts: torch.Tensor,
                user_feats: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        inputs = self._dropout(self._embed(inputs))
        # [B, 10, 20, 30, E]
        p_list = []
        for i, p in enumerate(inputs):
            # [10, 20, 30, E]
            r_list = []
            for j, r in enumerate(p):
                # [20, 30, E]
                # TODO 不加长度似乎效果更好？
                # h_n = self._w_rnn(r, sent_lengths[i, j])
                h_n = self._w_rnn(r)
                # [20, H]
                r_list.append(h_n.unsqueeze(0))

            r_batch = torch.cat(r_list, dim=0)
            # [10, 20, H]
            # h_n = self._s_rnn(r_batch, sent_counts[i])
            h_n = self._s_rnn(r_batch)
            # [10, H]
            p_list.append(h_n.unsqueeze(0))

        p_batch = torch.cat(p_list, dim=0)
        # [B, 10, H]
        if self._add_user:
            p_batch = torch.cat([p_batch, F.normalize(user_feats) * self.user_feats_weights], -1)
        # [B, 10, H+F]
        r_stars = self._r_fc(p_batch).squeeze()
        # [B, 10]
        h_n = self._r_rnn(p_batch)
        # [B, H]
        p_stars = self._p_fc(h_n).squeeze()
        # [B]
        return p_stars, r_stars
