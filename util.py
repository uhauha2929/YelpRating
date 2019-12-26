# -*- coding: utf-8 -*-
# @Time    : 2019/12/26 9:39
# @Author  : uhauha2929


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


