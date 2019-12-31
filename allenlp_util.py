# -*- coding: utf-8 -*-
# @Time    : 2019/12/27 19:14
# @Author  : uhauha2929
from typing import Dict, Callable

from allennlp.data import Token, Instance, TokenIndexer
from allennlp.data.fields import TextField


def text_to_instance(text: str,
                     word_tokenizer: Callable,
                     token_indexers: Dict[str, TokenIndexer],
                     field_name: str = 'tokens',
                     lowercase=True):
    tokens = [Token(word)
              for word in word_tokenizer(text.lower() if lowercase else text)]
    field = TextField(tokens, token_indexers)
    fields = {field_name: field}
    instance = Instance(fields)
    return instance
