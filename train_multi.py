import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn

from dataset import ProductDataset
from models.cnn_gru import MultiHierarchicalCnnBiGRU
from shared import *
from models.h_gru import MultiHierarchicalGRU, MultiHierarchicalBiGRU

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
regression = False

regress_criterion = nn.MSELoss().to(device)
classify_criterion = nn.CrossEntropyLoss().to(device)


def train(train_loader, model, optimizer):
    model.train()
    train_loss = []
    epoch_loss = 0
    bar = tqdm(total=len(train_loader))
    for b_id, (X, Y, R) in enumerate(train_loader, 1):
        X = X.to(device)
        Y = Y.to(device)
        R = R.to(device)
        optimizer.zero_grad()
        p_stars, r_stars = model(X)
        if regression:
            p_loss = regress_criterion(p_stars.squeeze(), Y.squeeze())
        else:
            p_loss = classify_criterion(p_stars.squeeze(), Y.squeeze())
        r_loss = regress_criterion(r_stars.squeeze(), R.squeeze())
        loss = p_loss + r_loss
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
    correct, total = 0, 0
    pred, target = [], []
    with torch.no_grad():
        for i, (X, Y, R) in enumerate(val_loader):
            X = X.to(device)
            Y = Y.to(device)
            R = R.to(device)
            p_stars, r_stars = model(X)
            if regression:
                p_loss = regress_criterion(p_stars.squeeze(), Y.squeeze())
            else:
                p_loss = classify_criterion(p_stars.squeeze(), Y.squeeze())
            r_loss = regress_criterion(r_stars.squeeze(), R.squeeze())
            loss = p_loss + r_loss
            epoch_loss += loss
            if regression:
                pred.append(p_stars.cpu().numpy().reshape(-1))
                target.append(Y.cpu().numpy().reshape(-1))
            else:
                correct += (torch.max(p_stars, -1)[1].view(-1) == Y.squeeze()).float().sum()
                total += Y.size(0)
    if regression:
        metric = mean_squared_error(np.concatenate(target), np.concatenate(pred))
    else:
        metric = correct.item() / total
    model.train()
    return epoch_loss.item() / len(val_loader), metric


def main():
    train_data = ProductDataset('data/old/products_train.txt',
                                'data/old/reviews_train.txt',
                                'data/old/vocab.json')

    val_data = ProductDataset('data/old/products_test.txt',
                              'data/old/reviews_test.txt',
                              'data/old/vocab.json')

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size)

    # model = MultiHierarchicalBiGRU(vocab_size + 2, emb_dim, hid_dim, regression).to(device)
    model = MultiHierarchicalCnnBiGRU(vocab_size + 2, emb_dim, hid_dim, regress=regression).to(device)
    model.embedding.weight.data.copy_(torch.Tensor(np.load('data/old/embedding_200.npy')))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = np.Inf if regression else 0
    for i in range(1, epoch + 1):
        train_loss = train(train_loader, model, optimizer)
        val_loss, val_acc = evaluate(model, val_loader)
        if val_acc < best_acc if regression else val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), '{}_{}.pt'.format(model.name, regression))
        print('| epoch: {:02} | train Loss: {:.3f} | val Loss: {:.3f} | val acc: {:.3f}'
              .format(i, train_loss[-1], val_loss, val_acc))


if __name__ == '__main__':
    main()
