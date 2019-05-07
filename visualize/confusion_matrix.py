# -*- coding: utf-8 -*-
# @Time    : 2019/5/6 14:13
# @Author  : uhauha2929
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(mat, labels):
    plt.imshow(mat, interpolation='nearest', cmap='cividis')
    plt.title('Confusion Matrix')
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    confusion_mat = np.array(
        [[2, 6, 1, 0, 0, 0, 0, 0, 0],
         [0, 19, 16, 2, 0, 0, 0, 0, 0],
         [0, 4, 37, 16, 5, 0, 1, 0, 0],
         [0, 0, 22, 53, 29, 3, 0, 0, 0],
         [0, 0, 1, 31, 71, 24, 7, 0, 0],
         [0, 0, 0, 1, 20, 00, 37, 3, 0],
         [0, 0, 0, 0, 3, 36, 129, 34, 0],
         [0, 0, 0, 0, 0, 1, 36, 112, 9],
         [0, 0, 0, 0, 0, 1, 1, 25, 108]]
    )
    labels = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    confusion_mat = confusion_mat.astype('float')
    confusion_mat /= confusion_mat.sum(-1).reshape((-1, 1)).repeat(9, -1)
    plot_confusion_matrix(confusion_mat, labels)
