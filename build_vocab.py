# -*- coding: utf-8 -*-
# @Time    : 2019/9/29 19:39
# @Author  : uhauha2929
import json
import os
from collections import Counter
from pathlib import Path
from typing import List

from allennlp.data.tokenizers.word_splitter import WordSplitter, JustSpacesWordSplitter

PADDING = '@@padding@@'
UNKNOWN = '@@unknown@@'
PADDING_CHAR = '∷'
UNKNOWN_CHAR = '※'


class Vocabulary(object):

    def __init__(self,
                 vocab_size: int = 20000,
                 vocab_file: str = './vocab/vocab.txt',
                 word_splitter: WordSplitter = JustSpacesWordSplitter()):
        self.vocab_size = vocab_size
        self.vocab_file = vocab_file

        self.word_splitter = word_splitter

        self.word_index, self.index_word = None, None

        if os.path.exists(vocab_file):
            self.load_vocab_from_txt(vocab_file)

    def load_vocab_from_txt(self, vocab_file: str):
        self.word_index = {}
        with open(vocab_file, 'rt') as v:
            for line in v:
                word = line.strip().lower()
                self.word_index[word] = len(self.word_index)

        self.vocab_size = len(self.word_index)
        self.index_word = {index: word for word, index in self.word_index.items()}

    @DeprecationWarning
    def get_char_index(self, max_char_size: int = 100):
        if self.word_index is None:
            raise Exception('vocab is empty!')
        char_counter = Counter()
        for word in self.word_index.keys():
            char_counter.update(word)
        char_index = {PADDING_CHAR: 0, UNKNOWN_CHAR: 1}
        for char, cnt in char_counter.most_common(max_char_size):
            char_index[char] = len(char_index)
        return char_index

    def build_vocabulary(self, sentences: List[str]):
        word_counter = Counter()
        for sentence in sentences:
            tokens = self.word_splitter.split_words(sentence)
            word_counter.update((token.text for token in tokens))

        words = [PADDING, UNKNOWN]

        Path(os.path.dirname(self.vocab_file)).mkdir(parents=True, exist_ok=True)

        with open(self.vocab_file, 'wt') as v:
            words.extend([word for word, _ in word_counter.most_common(self.vocab_size)])
            for word in words:
                v.write(word + '\n')

        self.load_vocab_from_txt(self.vocab_file)


# -----------------test----------------------
def test_build_vocab():
    vocab = Vocabulary(20000)
    texts = []
    with open('./data/reviews_train.txt', 'rt') as r:
        for line in r:
            review = json.loads(line)
            texts.append(review['text'])
    vocab.build_vocabulary(texts)
    print(vocab.vocab_size)
