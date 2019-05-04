# -*- coding: utf-8 -*-
# @Time    : 2019/5/3 21:35
# @Author  : uhauha2929
import re
import numpy as np
import visdom


def get_loss_from_file(file_path):
    pattern = re.compile(r'\d+\.\d\d\d\d')
    loss_list = []
    with open(file_path, 'rt') as f:
        for line in f:
            loss = pattern.findall(line)
            loss = map(float, loss)
            loss_list.extend(loss)

    print(len(loss_list))
    return loss_list


def main():
    h_bigru = get_loss_from_file('h_gru.out')
    h_bigru_multi = get_loss_from_file('h_gru_multi.out')
    h_bigru_multi_user = get_loss_from_file('h_gru_multi_user.out')

    viz = visdom.Visdom()
    viz.line(
        X=np.column_stack((
            np.arange(0, 5020),
            np.arange(0, 5020),
            np.arange(0, 5020)
        )),
        Y=np.column_stack((
            np.array(h_bigru_multi_user),
            np.array(h_bigru_multi),
            np.array(h_bigru)
        )),
        opts={
            'dash': np.array(['solid', 'dash', 'dashdot']),
            'title': '训练损失',
            'legend': ['h_bigru_multi_user', 'h_bigru_multi', 'h_bigru'],
            'xlabel': 'batch',
            'ylabel': 'loss',
            'width': 500,
            'height': 350
        })


if __name__ == '__main__':
    main()
