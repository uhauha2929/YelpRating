# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 19:33
# @Author  : uhauha2929
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from shared import *


def collate_fn(data):
    def merge(sequences):
        length = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        padded_seqs = torch.zeros(len(sequences), max(length), dtype=torch.long)
        for i, seq in enumerate(sequences):
            end = length[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, length

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    seqs, stars = zip(*data)
    # merge sequences (from tuple of 1D tensor to 2D tensor)
    seqs, length = merge(seqs)
    stars = torch.LongTensor(stars).squeeze()
    return seqs, length, stars


class ProductDataset(Dataset):

    def __init__(self,
                 products_path: str,
                 reviews_path: str,
                 vocab_path: str,
                 max_length: 200):

        self._max_length = max_length

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

    def __len__(self):
        return len(self._products)

    def __getitem__(self, index):
        product = self._products[index]
        review_ids = product['review_ids']

        word_id_list = []
        for r_id in review_ids:
            review = self._reviews[r_id]
            for word in review['text'].split():
                if word in self._word2id:
                    if len(word_id_list) == self._max_length:
                        break
                    word_id_list.append(self._word2id[word])

            if len(word_id_list) == self._max_length:
                break

        product_stars = torch.tensor(self._stars2label[product['stars']], dtype=torch.long)

        return torch.LongTensor(word_id_list), product_stars


class ProductUserDataset(Dataset):
    def __init__(self,
                 products_path: str,
                 reviews_path: str,
                 vocab_path: str,
                 user_feats_path: str,
                 num_reviews: int = 10,  # 每个产品评论个数
                 num_sentences: int = 20,  # 每个评论句子个数
                 max_sentence_length: int = 30,
                 regress: bool = False):  # 每个句子长度

        self._num_reviews = num_reviews
        self._num_sentences = num_sentences
        self._max_sentence_length = max_sentence_length
        self._regress = regress

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

        product_reviews_stars = torch.zeros([self._num_reviews])

        sent_lengths = torch.zeros([self._num_reviews, self._num_sentences], dtype=torch.long)
        sent_counts = torch.zeros([self._num_reviews], dtype=torch.long)

        review_user_feats = []

        for i, review_id in enumerate(review_ids[:self._num_reviews]):
            review = self._reviews[review_id]
            review_user_feats.append(self._user_feats[review['user_id']])

            product_reviews_stars[i] = int(review['stars'])
            text = review['text']
            sentences = text.split("\n")  # 句子按照\n分割
            sentences = sentences[:self._num_sentences]

            sent_counts[i] = len(sentences)

            for j, sentence in enumerate(sentences):
                words = sentence.split()  # 单词按照空格分割
                words = words[:self._max_sentence_length]
                sent_lengths[i, j] = len(words)
                for k, word in enumerate(words):
                    product_tensor[i, j, k] = self._word2id.get(word, self._word2id[unk])

        review_user_feats = torch.FloatTensor(review_user_feats)

        if self._regress:
            product_stars = product_reviews_stars.mean()
        else:
            product_stars = torch.tensor(self._stars2label[product['stars']], dtype=torch.long)

        output_dict = {"product": product_tensor,
                       "product_stars": product_stars,
                       "review_stars": product_reviews_stars,
                       "user": review_user_feats,
                       "sent_length": sent_lengths,
                       "sent_count": sent_counts}

        return output_dict

    def __len__(self):
        return len(self._products)
