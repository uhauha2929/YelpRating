# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 19:33
# @Author  : uhauha2929
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from shared import *


class ProductDataset(Dataset):
    def __init__(self,
                 products_path: str,
                 reviews_path: str,
                 vocab_path: str,
                 num_reviews: int = 10,  # 每个产品评论个数
                 num_sentences: int = 20,  # 每个评论句子个数
                 max_sentence_length: int = 30):  # 每个句子长度

        self._num_reviews = num_reviews
        self._num_sentences = num_sentences
        self._max_sentence_length = max_sentence_length

        with open(vocab_path, 'rt') as v:
            self._word2id = json.load(v)

        reviews = {}
        with open(reviews_path, 'rt') as r:
            for line in r:
                review = json.loads(line)
                reviews[review['review_id']] = review
        self._reviews = reviews

        products = []
        with open(products_path, 'rt') as p:
            for line in p:
                products.append(json.loads(line))
        self._products = products

        self._stars2label = {1.0: 0, 1.5: 1, 2.0: 2, 2.5: 3, 3.0: 4,
                             3.5: 5, 4.0: 6, 4.5: 7, 5.0: 8}

    def __getitem__(self, index):
        product_tensor = torch.zeros([self._num_reviews,
                                      self._num_sentences,
                                      self._max_sentence_length],
                                     dtype=torch.long)
        product = self._products[index]
        review_ids = product['review_ids']

        product_stars = torch.LongTensor([self._stars2label[product['stars']]])

        product_reviews_stars = torch.zeros([self._num_reviews])

        for i, review_id in enumerate(review_ids[:self._num_reviews]):
            review = self._reviews[review_id]
            product_reviews_stars[i] = int(review['stars'])
            text = review['text']
            sentences = text.split("\n")  # 句子按照\n分割
            for j, sentence in enumerate(sentences[:self._num_sentences]):
                words = sentence.split()  # 单词按照空格分割
                for k, word in enumerate(words[:self._max_sentence_length]):
                    product_tensor[i, j, k] = self._word2id.get(word, self._word2id[unk])

        return product_tensor, product_stars, product_reviews_stars

    def __len__(self):
        return len(self._products)


class ProductUserDataset(Dataset):
    def __init__(self,
                 products_path: str,
                 reviews_path: str,
                 vocab_path: str,
                 user_feats_path: str,
                 num_reviews: int = 10,  # 每个产品评论个数
                 num_sentences: int = 20,  # 每个评论句子个数
                 max_sentence_length: int = 30):  # 每个句子长度

        self._num_reviews = num_reviews
        self._num_sentences = num_sentences
        self._max_sentence_length = max_sentence_length

        with open(vocab_path, 'rt') as v:
            self._word2id = json.load(v)

        reviews = {}
        with open(reviews_path, 'rt') as r:
            for line in r:
                review = json.loads(line)
                reviews[review['review_id']] = review
        self._reviews = reviews

        products = []
        with open(products_path, 'rt') as p:
            for line in p:
                products.append(json.loads(line))
        self._products = products

        with open(user_feats_path, 'rt') as u:
            self._user_feats = json.load(u)

        self._stars2label = {1.0: 0, 1.5: 1, 2.0: 2, 2.5: 3, 3.0: 4,
                             3.5: 5, 4.0: 6, 4.5: 7, 5.0: 8}

    def __getitem__(self, index):
        product_tensor = torch.zeros([self._num_reviews,
                                      self._num_sentences,
                                      self._max_sentence_length],
                                     dtype=torch.long)
        product = self._products[index]
        review_ids = product['review_ids']

        product_stars = torch.LongTensor([self._stars2label[product['stars']]])

        product_reviews_stars = torch.zeros([self._num_reviews])

        review_user_feats = []

        for i, review_id in enumerate(review_ids[:self._num_reviews]):
            review = self._reviews[review_id]
            review_user_feats.append(self._user_feats[review['user_id']])

            product_reviews_stars[i] = int(review['stars'])
            text = review['text']
            sentences = text.split("\n")  # 句子按照\n分割
            for j, sentence in enumerate(sentences[:self._num_sentences]):
                words = sentence.split()  # 单词按照空格分割
                for k, word in enumerate(words[:self._max_sentence_length]):
                    product_tensor[i, j, k] = self._word2id.get(word, self._word2id[unk])

        review_user_feats = torch.FloatTensor(review_user_feats)
        return product_tensor, product_stars, product_reviews_stars, review_user_feats

    def __len__(self):
        return len(self._products)
