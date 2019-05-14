# -*- coding: utf-8 -*-
# @Time    : 2019/4/30 12:49
# @Author  : uhauha2929
from typing import List

import visdom
import numpy as np


def plot_data_count(viz: visdom.Visdom,
                    train_count: List[int],
                    test_count: List[int]):
    assert len(train_count) == len(test_count)
    n_class = len(train_count)
    train_count = np.array(train_count).reshape((n_class, 1))
    test_count = np.array(test_count).reshape((n_class, 1))
    viz.bar(
        X=np.concatenate((train_count, test_count), axis=-1),
        opts={
            'stacked': True,
            'legend': ['train', 'test'],
            'rownames': ['1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5'],
            'title': '数据集统计',
            'xlabel': 'stars',
            'ylabel': 'count'
        }
    )


def plot_accuracy(viz: visdom.Visdom):
    h_bigru_multi_user = [0.463, 0.527, 0.558, 0.597, 0.582, 0.595, 0.526, 0.535, 0.595, 0.619,
                          0.610, 0.625, 0.615, 0.621, 0.612, 0.570, 0.626, 0.608, 0.586, 0.559]
    h_bigru_multi = [0.468, 0.540, 0.459, 0.572, 0.583, 0.592, 0.552, 0.526, 0.570, 0.599,
                     0.601, 0.594, 0.606, 0.586, 0.579, 0.604, 0.576, 0.602, 0.590, 0.589]
    h_bigru = [0.339, 0.454, 0.534, 0.502, 0.546, 0.498, 0.531, 0.557, 0.529, 0.518,
               0.560, 0.556, 0.550, 0.546, 0.562, 0.506, 0.444, 0.479, 0.498, 0.512]
    viz.line(
        X=np.column_stack((
            np.arange(0, 20),
            np.arange(0, 20),
            np.arange(0, 20),
        )),
        Y=np.column_stack((
            np.array(h_bigru_multi_user),
            np.array(h_bigru_multi),
            np.array(h_bigru)
        )),
        opts={
            'dash': np.array(['solid', 'dash', 'dashdot']),
            'title': '准确率比较',
            'legend': ['H-GRU-Multi-User', 'H-GRU-Multi', 'H-GRU'],
            'xlabel': 'epoch',
            'ylabel': 'accuracy',
            'width': 500,
            'height': 350
        })


def main():
    # 训练集和测试集的个数统计
    train = [49, 113, 235, 406, 539, 661, 757, 702, 559]
    test = [9, 37, 63, 107, 134, 161, 202, 158, 135]
    vis = visdom.Visdom()
    plot_data_count(vis, train, test)
    plot_accuracy(vis)


if __name__ == '__main__':
    main()
