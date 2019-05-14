import numpy as np
import visdom
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn

from dataset import ProductUserDataset
from models.cnn_gru import CnnMulti2GruUser
from shared import *
from models.h_gru import Multi3GruUser
from functools import partial
from models.loss import focal_loss
from visualize.loss import LossPainter, LinePainter

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

regress_criterion = nn.MSELoss().to(device)
classify_criterion = nn.CrossEntropyLoss().to(device)


# weight = torch.FloatTensor([0.99, 0.97, 0.94, 0.90, 0.87, 0.84, 0.81, 0.83, 0.86])
# weight = torch.FloatTensor([1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.4, 1.5])
# classify_criterion = partial(focal_loss, device=device, weight=weight)


def train(train_loader, model, optimizer):
    model.train()
    train_loss = []
    epoch_loss = 0
    bar = tqdm(total=len(train_loader))
    for b_id, output_dict in enumerate(train_loader, 1):
        product = output_dict['product'].to(device)
        product_stars = output_dict['product_stars'].to(device)
        review_stars = output_dict['review_stars'].to(device)
        user = output_dict['user'].to(device)
        sent_lengths = output_dict['sent_length'].to(device)
        sent_counts = output_dict['sent_count'].to(device)

        optimizer.zero_grad()
        p_stars, r_stars = model(product, sent_lengths, sent_counts, user)
        if regression:
            p_loss = regress_criterion(p_stars, product_stars)
        else:
            p_loss = classify_criterion(p_stars, product_stars)

        r_loss = regress_criterion(r_stars, review_stars)
        loss = p_loss + r_loss

        # loss = p_loss

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
        for i, output_dict in enumerate(val_loader):
            product = output_dict['product'].to(device)
            product_stars = output_dict['product_stars'].to(device)
            review_stars = output_dict['review_stars'].to(device)
            user = output_dict['user'].to(device)
            sent_lengths = output_dict['sent_length'].to(device)
            sent_counts = output_dict['sent_count'].to(device)

            p_stars, r_stars = model(product, sent_lengths, sent_counts, user)
            if regression:
                p_loss = regress_criterion(p_stars, product_stars)
            else:
                p_loss = classify_criterion(p_stars, product_stars)

            r_loss = regress_criterion(r_stars, review_stars)
            loss = p_loss + r_loss

            # loss = p_loss

            epoch_loss += loss

            pred.append(p_stars.max(-1)[1].cpu().numpy().reshape(-1))
            target.append(product_stars.cpu().numpy().reshape(-1))

    metric = {}
    y_true, y_pred = np.concatenate(target), np.concatenate(pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    metric['acc'] = accuracy_score(y_true, y_pred)
    metric['precision'] = precision
    metric['recall'] = recall
    metric['fscore'] = fscore

    if regression:
        metric['mse'] = mean_squared_error(np.concatenate(target), np.concatenate(pred))

    return epoch_loss.item() / len(val_loader), metric


def main():
    train_data = ProductUserDataset('data/old/products_train.txt',
                                    'data/old/reviews_train.txt',
                                    'data/old/vocab.json',
                                    'data/old/users_feats_scaled.json',
                                    regress=regression)

    val_data = ProductUserDataset('data/old/products_test.txt',
                                  'data/old/reviews_test.txt',
                                  'data/old/vocab.json',
                                  'data/old/users_feats_scaled.json',
                                  regress=regression)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size)

    model = Multi3GruUser(vocab_size, emb_dim, hid_dim, regress=regression, add_user=True).to(device)
    # model = CnnMulti2GruUser(vocab_size, emb_dim, hid_dim, regress=regression, add_user=False).to(device)

    model.load_embed_matrix(torch.Tensor(np.load('data/old/embedding_200.npy')))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # viz = visdom.Visdom()
    # loss_painter = LossPainter(viz)
    # acc_painter = LinePainter(viz, '准确率')
    # val_painter = LinePainter(viz, '测试损失')

    best_acc = np.inf if regression else -np.inf

    for i in range(1, epoch + 1):
        train_loss = train(train_loader, model, optimizer)
        val_loss, metric = evaluate(model, val_loader)

        print('| epoch: {:02} | train Loss: {:.3f} | val Loss: {:.3f}'
              .format(i, train_loss[-1], val_loss))
        print(metric)

        if regression:
            if metric['mse'] < best_acc:
                best_acc = metric['mse']
                torch.save(model.state_dict(), 'best_regression.pt')
        else:
            if metric['acc'] > best_acc:
                best_acc = metric['acc']
                torch.save(model.state_dict(), 'best_classification.pt')

        # loss_painter.update_epoch(train_loss)
        # acc_painter.update(val_acc)
        # val_painter.update(val_loss)


if __name__ == '__main__':
    main()
