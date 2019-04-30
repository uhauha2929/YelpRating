# -*- coding: utf-8 -*-
# @Time    : 2019/4/30 13:50
# @Author  : uhauha2929
from typing import List
import numpy as np

import visdom


class LinePainter(object):
    def __init__(self, viz: visdom.Visdom, name: str):
        self._viz = viz
        self._win = None
        self._name = name
        self._end_axis = 0

    def update(self, val: float):
        x = np.array([self._end_axis])
        y = np.array([val])
        if self._win is None:
            self._win = self._viz.line(
                X=x,
                Y=y,
                opts={
                    'title': self._name,
                }
            )
        else:
            self._win = self._viz.line(
                X=x,
                Y=y,
                win=self._win,
                update='append')

        self._end_axis += 1


class LossPainter(LinePainter):

    def __init__(self, viz: visdom.Visdom, name: str = '训练损失'):
        super(LossPainter, self).__init__(viz, name)

    def update_epoch(self, loss: List[float]):
        if self._win is None:
            self._win = self._viz.line(
                X=np.arange(0, len(loss)),
                Y=np.array(loss),
                opts={
                    'title': self._name
                }
            )
        else:
            self._win = self._viz.line(
                X=np.arange(self._end_axis, self._end_axis + len(loss)),
                Y=np.array(loss),
                win=self._win,
                update='append')

        self._end_axis += len(loss)


if __name__ == '__main__':
    vis = visdom.Visdom()
    painter = LossPainter(vis)
    painter.update_epoch([10, 9, 8, 7, 6])
    painter.update_epoch([5, 4, 3, 2, 1])

    val_painter = LinePainter(vis, 'loss')
    acc_painter = LinePainter(vis, 'acc')
    for i in range(1, 100):
        val_painter.update(100 / i)
        acc_painter.update(np.sqrt(i / 100))
