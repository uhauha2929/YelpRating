# -*- coding: utf-8 -*-
# @Time    : 2019/12/26 9:39
# @Author  : uhauha2929
import torch

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


conf = Config(
    embedding_size=200,

    hidden_size=128,
    learning_rate=1e-3,

    batch_size=64,
    epoch=50
)

conf_bert = Config(
    pretrained_data_dir='/home/yzhao/data/bert/bert-base-uncased',

    hidden_dim=128,
    output_dim=9,
    dropout=0.2,
    user_feats_dim=20,

    learning_rate=1e-3,

    batch_size=64,
    epoch=50
)

conf_dan = Config(
    embedding_size=200,

    hidden_size=128,
    learning_rate=1e-3,

    batch_size=64,
    epoch=50
)
