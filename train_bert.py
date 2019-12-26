import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn

from config import conf_bert
from dataset_bert import ProductUserDatasetBERT
from build_vocab import Vocabulary
from models.hierarchical_bert import HierarchicalJointModelBERT

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

regress_criterion = nn.MSELoss().to(DEVICE)
classify_criterion = nn.CrossEntropyLoss().to(DEVICE)


def train(train_loader, model, optimizer):
    model.train()
    train_loss = []
    epoch_loss = 0
    bar = tqdm(total=len(train_loader))
    for b_id, output_dict in enumerate(train_loader, 1):
        product = output_dict['product'].to(DEVICE)
        product_star = output_dict['product_star'].to(DEVICE)
        review_stars = output_dict['review_stars'].to(DEVICE)
        user_features = output_dict['user_features'].to(DEVICE)

        optimizer.zero_grad()
        p_stars, r_stars = model(product, user_features)
        p_loss = classify_criterion(p_stars, product_star)

        r_loss = regress_criterion(r_stars, review_stars)
        loss = p_loss + r_loss

        loss.backward()
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
            product = output_dict['product'].to(DEVICE)
            product_star = output_dict['product_star'].to(DEVICE)
            review_stars = output_dict['review_stars'].to(DEVICE)
            user_features = output_dict['user_features'].to(DEVICE)

            p_stars, r_stars = model(product, user_features)
            p_loss = classify_criterion(p_stars, product_star)

            r_loss = regress_criterion(r_stars, review_stars)
            loss = p_loss + r_loss

            epoch_loss += loss

            pred.append(p_stars.max(-1)[1].cpu().numpy().reshape(-1))
            target.append(product_star.cpu().numpy().reshape(-1))
    metric = {}
    y_true, y_pred = np.concatenate(target), np.concatenate(pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    metric['acc'] = accuracy_score(y_true, y_pred)
    metric['precision'] = precision
    metric['recall'] = recall
    metric['fscore'] = fscore

    return epoch_loss.item() / len(val_loader), metric


def main():
    train_data = ProductUserDatasetBERT('data/products_train.txt',
                                        'data/reviews_train.txt',
                                        'data/users_feats.json')

    val_data = ProductUserDatasetBERT('data/products_test.txt',
                                      'data/reviews_test.txt',
                                      'data/users_feats.json')

    train_loader = DataLoader(dataset=train_data, batch_size=conf_bert.batch_size)
    val_loader = DataLoader(dataset=val_data, batch_size=conf_bert.batch_size)

    model = HierarchicalJointModelBERT(conf_bert.hidden_size).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf_bert.learning_rate)

    best_acc = -np.inf

    for i in range(1, conf_bert.epoch + 1):
        train_loss = train(train_loader, model, optimizer)
        val_loss, metric = evaluate(model, val_loader)

        print('| epoch: {:02} | train Loss: {:.3f} | val Loss: {:.3f}'
              .format(i, train_loss[-1], val_loss))
        print(metric)

        if metric['acc'] > best_acc:
            best_acc = metric['acc']
            torch.save(model.state_dict(), 'bert.pt')


if __name__ == '__main__':
    main()
