# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 20:04
# @Author  : uhauha2929

pad = "<pad>"
unk = "<unk>"

vocab_size = 50000 + 2  # for pad and unk

regression = True

emb_dim = 200
hid_dim = 256

grad_clip = 10
learning_rate = 1e-3

epoch = 50
batch_size = 32
