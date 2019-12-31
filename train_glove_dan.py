from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn

from build_vocab_embedding import Vocabulary
from config import conf_dan, DEVICE
from dataset import ProductUserDataset
from models.hierarchical_dan import HierarchicalJointModelDAN

log_dir = 'log_glove_dan'
Path(log_dir).mkdir(parents=True, exist_ok=True)

regress_criterion = nn.MSELoss().to(DEVICE)
classify_criterion = nn.CrossEntropyLoss().to(DEVICE)


def train(train_loader, model, optimizer):
    model.train()
    total_losses, p_losses, r_losses = [], [], []
    total_loss_sum, p_loss_sum, r_loss_sum = 0, 0, 0
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

        p_loss_sum += p_loss.item()
        r_loss_sum += r_loss.item()
        total_loss_sum += loss.item()
        bar.update()
        bar.set_description('total_loss:{:.4f}|p_loss:{:.4f}|r_loss:{:.4f}'
                            .format(total_loss_sum / b_id, p_loss_sum / b_id, r_loss_sum / b_id))
        total_losses.append(total_loss_sum / b_id)
        p_losses.append(p_loss_sum / b_id)
        r_losses.append(r_loss_sum / b_id)

    bar.close()
    return {'total_loss': total_losses, 'p_loss': p_losses, 'r_loss': r_losses}


def evaluate(model, val_loader):
    model.eval()
    epoch_loss = 0
    pred, target = [], []
    with torch.no_grad():
        for i, output_dict in enumerate(tqdm(val_loader)):
            product = output_dict['product'].to(DEVICE)
            product_star = output_dict['product_star'].to(DEVICE)
            review_stars = output_dict['review_stars'].to(DEVICE)
            user_features = output_dict['user_features'].to(DEVICE)

            p_stars, r_stars = model(product, user_features)
            p_loss = classify_criterion(p_stars, product_star)
            r_loss = regress_criterion(r_stars, review_stars)
            loss = p_loss + r_loss

            epoch_loss += loss.item()

            pred.append(p_stars.max(-1)[1].cpu().numpy().reshape(-1))
            target.append(product_star.cpu().numpy().reshape(-1))
    metric = {}
    y_true, y_pred = np.concatenate(target), np.concatenate(pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    metric['accuracy'] = accuracy_score(y_true, y_pred)
    metric['precision'] = precision
    metric['recall'] = recall
    metric['fscore'] = fscore
    metric['loss'] = epoch_loss / len(val_loader)
    return metric


def main():
    vocab = Vocabulary()
    train_data = ProductUserDataset(vocab, 'data/products_train.txt',
                                    'data/tokenized_reviews.txt',
                                    'data/users_feats.json')

    val_data = ProductUserDataset(vocab, 'data/products_test.txt',
                                  'data/tokenized_reviews.txt',
                                  'data/users_feats.json')

    train_loader = DataLoader(dataset=train_data, batch_size=conf_dan.batch_size)
    val_loader = DataLoader(dataset=val_data, batch_size=conf_dan.batch_size)

    model = HierarchicalJointModelDAN(vocab.vocab_size,
                                      conf_dan.embedding_size,
                                      conf_dan.hidden_size).to(DEVICE)
    model.load_word_embedding('./word_embedding_200.npy')
    optimizer = torch.optim.Adam(model.parameters(), lr=conf_dan.learning_rate)

    best_acc = -np.inf

    for i in range(1, conf_dan.epoch + 1):
        train_losses_dict = train(train_loader, model, optimizer)
        metric = evaluate(model, val_loader)

        np.save('{}/glove-dan-epoch-{:02}-loss'.format(log_dir, i), train_losses_dict)
        print(metric)

        if metric['accuracy'] > best_acc:
            best_acc = metric['accuracy']
            torch.save(model.state_dict(), '{}/glove_dan.pt'.format(log_dir))


if __name__ == '__main__':
    main()
