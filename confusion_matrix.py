# -*- coding: utf-8 -*-
# @Time    : 2019/5/6 13:12
# @Author  : uhauha2929
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from dataset import ProductUserDataset
from models.h_gru import Multi3GruUser
from shared import *
import numpy as np
import visdom

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def show_heat_map(model, val_loader):
    model.eval()
    pred, target = [], []
    with torch.no_grad():
        for i, output_dict in enumerate(val_loader):
            product = output_dict['product'].to(device)
            product_stars = output_dict['product_stars'].to(device)
            user = output_dict['user'].to(device)
            sent_lengths = output_dict['sent_length'].to(device)
            sent_counts = output_dict['sent_count'].to(device)

            p_stars, r_stars = model(product, sent_lengths, sent_counts, user)

            pred.append(p_stars.max(-1)[1].cpu().numpy().reshape(-1))
            target.append(product_stars.cpu().numpy().reshape(-1))

    y_true, y_pred = np.concatenate(target), np.concatenate(pred)
    confusion_mat = confusion_matrix(y_true, y_pred)
    print(confusion_mat)
    print(confusion_mat.sum())
    viz = visdom.Visdom()
    viz.heatmap(
        X=confusion_mat,
        opts={
            'columnnames': ['1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5'],
            'rownames': ['1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5'],
            'xtickstep': 0.5,
            'ytickstep': 0.5,
            'colormap': 'Electric',
            'title': '混淆矩阵可视化'
        }
    )


def main():
    val_data = ProductUserDataset('data/old/products_test.txt',
                                  'data/old/reviews_test.txt',
                                  'data/old/vocab.json',
                                  'data/old/users_feats_scaled.json',
                                  regress=regression)

    val_loader = DataLoader(dataset=val_data, batch_size=batch_size)
    model = Multi3GruUser(vocab_size, emb_dim, hid_dim, regress=regression, add_user=True).to(device)
    model.load_state_dict(torch.load('best_classification.pt'))
    show_heat_map(model, val_loader)


if __name__ == '__main__':
    main()
