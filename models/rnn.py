import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class SimpleRNN(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, hid_dim: int, regress: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * hid_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, 1 if regress else 9)
        )

    def load_embed_matrix(self, matrix: torch.Tensor):
        self.embedding.weight.data.copy_(matrix)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor):
        batch_size = inputs.size(0)
        inputs = inputs.view(batch_size, -1)
        inputs = self.embedding(inputs)
        packed_seq = pack_padded_sequence(inputs, lengths, batch_first=True)
        _, (h_n, _) = self.rnn(packed_seq)
        h_n = torch.cat([h_n[0], h_n[1]], -1)
        return F.elu(self.fc(h_n)).squeeze()