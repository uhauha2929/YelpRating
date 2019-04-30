# -*- coding: utf-8 -*-
# @Time    : 2019/4/30 12:49
# @Author  : uhauha2929
from typing import List

import visdom
import numpy as np


def plot_data(viz: visdom.Visdom,
              train_count: List[int],
              test_count: List[int]):
    assert len(train_count) == len(test_count)
    n_class = len(train_count)
    train_count = np.array(train_count).reshape((n_class, 1))
    test_count = np.array(test_count).reshape((n_class, 1))
    viz.bar(X=np.random.rand(n_class))
    viz.bar(
        X=np.concatenate((train_count, test_count), axis=-1),
        opts={
            'stacked': True,
            'legend': ['train', 'test'],
            'rownames': ['1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5'],
            'title': '数据集统计'
        }
    )


def main():
    # 训练集和测试集的个数统计
    train = [49, 113, 235, 406, 539, 661, 757, 702, 559]
    test = [9, 37, 63, 107, 134, 161, 202, 158, 135]
    vis = visdom.Visdom()
    plot_data(vis, train, test)


if __name__ == '__main__':
    main()
