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

from build_vocab_embedding import Vocabulary
from dataset import ProductDataset, collate_fn
from models.cnn import SimpleCNN
from models.dan import DANWrapper
from models.rnn import SimpleRNN

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
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
    vocab = Vocabulary()
    train_data = ProductDataset(vocab,
                                'data/products_train.txt',
                                'data/tokenized_reviews.txt', max_length=1000)

    val_data = ProductDataset(vocab,
                              'data/products_test.txt',
                              'data/tokenized_reviews.txt', max_length=1000)

    train_loader = DataLoader(dataset=train_data, batch_size=64, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_data, batch_size=64, collate_fn=collate_fn)

    # model = SimpleRNN(vocab.vocab_size, 200, 200).to(device)
    model = SimpleCNN(vocab.vocab_size, 200, 200).to(device)
    # model = DANWrapper(vocab.vocab_size, 200).to(device)

    model.load_embed_matrix(torch.Tensor(np.load('word_embedding_200.npy')))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i in range(1, 20 + 1):
        train_loss = train(train_loader, model, optimizer)
        val_loss, metric = evaluate(model, val_loader)
        print(train_loss[-1], val_loss)
        pprint(metric)


if __name__ == '__main__':
    main()
