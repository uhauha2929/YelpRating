# -*- coding: utf-8 -*-
# @Time    : 2019/5/6 14:13
# @Author  : uhauha2929
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


if __name__ == '__main__':
    confusion_mat = np.array(
        [[5, 4, 0, 0, 0, 0, 0, 0, 0],
         [1, 18, 17, 1, 0, 0, 0, 0, 0],
         [0, 6, 38, 14, 4, 1, 0, 0, 0],
         [0, 0, 28, 49, 28, 1, 1, 0, 0],
         [0, 0, 2, 27, 61, 35, 9, 0, 0],
         [0, 0, 0, 1, 18, 101, 39, 2, 0],
         [0, 0, 0, 1, 2, 32, 130, 36, 1],
         [0, 0, 0, 0, 0, 1, 25, 120, 12],
         [0, 0, 0, 0, 0, 0, 1, 17, 117]]
    )

    labels = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    sns.heatmap(confusion_mat, square=True, fmt='d', annot=True, cmap='cividis',
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('Confusion Matrix')
    plt.title('Confusion Matrix')
    plt.show()
