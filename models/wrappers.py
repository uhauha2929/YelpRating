# -*- coding: utf-8 -*-
# @Time    : 2019/4/27 16:53
# @Author  : uhauha2929
import torch
import torch.nn as nn


class GruWrapper(nn.Module):

    def __init__(self, gru: nn.GRU,
                 return_last: bool = False):
        super().__init__()
        self._gru = gru
        self._hid_dim = gru.hidden_size
        self._return_last = return_last
        self._batch_first = gru.batch_first
        self._bidirectional = gru.bidirectional

    def forward(self, inp: torch.Tensor,
                lengths: torch.Tensor = None):
        # inp: [batch, seq_len, emb_dim]
        outputs, h_n = self._gru(inp)
        # outputs: [batch, seq_len, hid_dim]
        if not self._return_last:
            return outputs
        if lengths is None:
            if self._bidirectional:
                h_n = h_n[0] + h_n[1]
            return h_n
        # mask: [batch, seq_len]
        # The insufficient length is filled with 0
        lengths = torch.LongTensor([i - 1 if i > 0 else 0 for i in lengths])
        if not self._batch_first:
            outputs = outputs.transpose(0, 1)
        forward_last = outputs[torch.arange(inp.size(0)), lengths, :]
        if self._bidirectional:
            forward_last = forward_last[:, :self._hid_dim] + forward_last[:, self._hid_dim:]
        return forward_last
