import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import conf_bert
from dataset_bert import ProductUserDatasetBERT
from models.hierarchical_bert import HierarchicalJointModelBERT
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


def show_heat_map(model, val_loader):
    model.eval()
    pred, target = [], []
    with torch.no_grad():
        for output_dict in tqdm(val_loader):
            product = output_dict['product'].to(device)
            product_star = output_dict['product_star'].to(device)
            user_features = output_dict['user_features'].to(device)

            p_stars, _ = model(product, user_features)

            pred.append(p_stars.max(-1)[1].cpu().numpy().reshape(-1))
            target.append(product_star.cpu().numpy().reshape(-1))

    y_true, y_pred = np.concatenate(target), np.concatenate(pred)
    confusion_mat = confusion_matrix(y_true, y_pred)
    print(confusion_mat)
    print(confusion_mat.sum())

    labels = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    sns.heatmap(confusion_mat,
                square=True, fmt='d', annot=True, cmap='cividis',
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.savefig('confusion_matrix')
    plt.title('confusion_matrix')
    plt.show()


def main():
    val_data = ProductUserDatasetBERT(conf_bert, 'data/products_test.txt',
                                      'data/tokenized_reviews.txt',
                                      'data/users_feats.json')

    val_loader = DataLoader(dataset=val_data, batch_size=conf_bert.batch_size)
    model = HierarchicalJointModelBERT(conf_bert).to(device)
    model.load_state_dict(torch.load('log_bert/bert.pt'))
    show_heat_map(model, val_loader)


if __name__ == '__main__':
    main()
