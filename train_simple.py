# -*- coding: utf-8 -*-
# @Time    : 2019/5/4 19:46
# @Author  : uhauha2929
from pprint import pprint

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn

from dataset import ProductDataset, collate_fn
from models.cnn import SimpleCNN
from models.rnn import SimpleRNN
from shared import *

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

classify_criterion = nn.CrossEntropyLoss().to(device)


def train(train_loader, model, optimizer):
    model.train()
    train_loss = []
    epoch_loss = 0
    bar = tqdm(total=len(train_loader))
    for b_id, (product, lengths, product_stars) in enumerate(train_loader, 1):
        product = product.to(device)
        product_stars = product_stars.to(device)
        optimizer.zero_grad()
        p_stars = model(product, lengths)
        loss = classify_criterion(p_stars, product_stars)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        epoch_loss += loss.item()
        bar.update()
        bar.set_description('current loss:{:.4f}'.format(epoch_loss / b_id))
        train_loss.append(epoch_loss / b_id)

    bar.close()
    return train_loss


def evaluate(model, val_loader):
    model.eval()
    epoch_loss = 0
    pred, target = [], []
    with torch.no_grad():
        for b_id, (product, lengths, product_stars) in enumerate(val_loader, 1):
            product = product.to(device)
            product_stars = product_stars.to(device)

            p_stars = model(product, lengths)
            loss = classify_criterion(p_stars, product_stars)

            epoch_loss += loss
            pred.append(p_stars.max(-1)[1].cpu().numpy().reshape(-1))
            target.append(product_stars.cpu().numpy().reshape(-1))

        metric = {}
        y_true, y_pred = np.concatenate(target), np.concatenate(pred)
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        metric['acc'] = (y_true == y_pred).sum() / len(y_true)
        metric['precision'] = precision
        metric['recall'] = recall
        metric['fscore'] = fscore

    return epoch_loss.item() / len(val_loader), metric


def main():
    train_data = ProductDataset('data/old/products_train.txt',
                                'data/old/reviews_train.txt',
                                'data/old/vocab.json', max_length=500)

    val_data = ProductDataset('data/old/products_test.txt',
                              'data/old/reviews_test.txt',
                              'data/old/vocab.json', max_length=500)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, collate_fn=collate_fn)

    # model = SimpleRNN(vocab_size, emb_dim, hid_dim).to(device)
    model = SimpleCNN(vocab_size, emb_dim, hid_dim).to(device)

    model.load_embed_matrix(torch.Tensor(np.load('data/old/embedding_200.npy')))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(1, epoch + 1):
        train_loss = train(train_loader, model, optimizer)
        val_loss, metric = evaluate(model, val_loader)
        print(train_loss[-1], val_loss)
        pprint(metric)


if __name__ == '__main__':
    main()
