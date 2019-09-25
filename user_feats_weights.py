# -*- coding: utf-8 -*-
# @Time    : 2019/5/13 21:14
# @Author  : uhauha2929
import torch
import torch.nn.functional as F
import visdom

from models.h_gru import Multi3GruUser
from shared import *

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    model = Multi3GruUser(vocab_size, emb_dim, hid_dim, add_user=True).to(device)
    model.load_state_dict(torch.load('best_classification.pt'))

    print(model.user_feats_weights)
    user_feats_weights = F.softmax(model.user_feats_weights, -1).detach().cpu().numpy()
    print(user_feats_weights)
    viz = visdom.Visdom()
    viz.bar(
        X=user_feats_weights,
        opts={
            'rownames': ['review_count', 'yelping_since', 'friends', 'useful', 'funny',
                         'cool', 'fans', 'elite', 'average_stars', 'compliment_hot',
                         'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list',
                         'compliment_note',
                         'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer',
                         'compliment_photos', ],
            'title': '用户特征权重分布',
            'xlabel': 'user features',
            'ylabel': 'weight',
            'width': 800,
            'height': 400,
            'marginbottom': 150,
            'rotation': 45
        }
    )
