# -*- coding: utf-8 -*-
# @Time    : 2019/9/29 19:39
# @Author  : uhauha2929
import numpy as np
import json
import os
from collections import Counter
from pathlib import Path
from typing import List

from allennlp.data.tokenizers.word_splitter import WordSplitter, JustSpacesWordSplitter

PADDING = '@@padding@@'
UNKNOWN = '@@unknown@@'

START = '@@start@@'
END = '@@end@@'

PADDING_CHAR = '∷'
UNKNOWN_CHAR = '※'


class Vocabulary(object):

    def __init__(self,
                 num_words: int = 20000,
                 vocab_file: str = './vocab/vocab.txt',
                 word_splitter: WordSplitter = JustSpacesWordSplitter()):

        self.num_words = num_words

        self.vocab_size = None

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
                if word:
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

        words = [PADDING, UNKNOWN, START, END]

        Path(os.path.dirname(self.vocab_file)).mkdir(parents=True, exist_ok=True)

        with open(self.vocab_file, 'wt') as v:
            words.extend([word for word, _ in word_counter.most_common(self.num_words)])
            for word in words:
                v.write(word + '\n')

        self.load_vocab_from_txt(self.vocab_file)


if __name__ == '__main__':
    vocab = Vocabulary(num_words=20000)
    texts = []
    with open('./data/tokenized_reviews.txt', 'rt') as r:
        for line in r:
            review = json.loads(line)
            texts.append(review['text'])
    vocab.build_vocabulary(texts)
    print(vocab.vocab_size)

    glove_dir = '/home/yzhao/data'
    embedding_dim = 200

    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.{}d.txt'.format(embedding_dim)), 'rt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((vocab.vocab_size, embedding_dim))
    for word, i in vocab.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i < vocab.vocab_size:
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

    print(embedding_matrix.shape)

    np.save('word_embedding_{}'.format(embedding_dim), embedding_matrix)
