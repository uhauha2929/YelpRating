# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 19:33
# @Author  : uhauha2929
import json
import torch
from torch.utils.data import Dataset

from build_vocab_embedding import Vocabulary, UNKNOWN, PADDING


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
                 vocab: Vocabulary,
                 products_path: str,
                 reviews_path: str,
                 max_length: 200):

        self._max_length = max_length
        self.vocab = vocab

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

    def __len__(self):
        return len(self._products)

    def __getitem__(self, index):
        product = self._products[index]
        review_ids = product['review_ids']

        word_id_list = []
        for r_id in review_ids:
            review = self._reviews[r_id]
            for word in review['text'].split():
                word = word.lower()
                if word in self.vocab.word_index:
                    if len(word_id_list) == self._max_length:
                        break
                    word_id_list.append(self.vocab.word_index[word])

            if len(word_id_list) == self._max_length:
                break

        product_stars = torch.tensor(int((product['stars'] - 1) / 0.5), dtype=torch.long)

        return torch.LongTensor(word_id_list), product_stars


class ProductUserDataset(Dataset):
    def __init__(self,
                 vocab: Vocabulary,
                 products_path: str,
                 reviews_path: str,
                 user_feats_path: str,
                 num_reviews: int = 10,
                 num_sentences: int = 20,
                 max_sequence_length: int = 30):

        self.num_reviews = num_reviews
        self.num_sentences = num_sentences
        self.max_sequence_length = max_sequence_length

        self.vocab = vocab

        self.review_dict = {}
        with open(reviews_path, 'rt') as r:
            for line in r:
                review = json.loads(line)
                self.review_dict[review['review_id']] = review

        self.products = []
        with open(products_path, 'rt') as p:
            for line in p:
                self.products.append(json.loads(line))

        with open(user_feats_path, 'rt') as u:
            self._user_feats = json.load(u)

    def __getitem__(self, index):
        product_tensor = torch.full([self.num_reviews,
                                     self.num_sentences,
                                     self.max_sequence_length],
                                    self.vocab.word_index[PADDING],
                                    dtype=torch.long)
        product = self.products[index]
        review_ids = product['review_ids']

        review_stars = torch.zeros([self.num_reviews])

        user_features = []

        for i, review_id in enumerate(review_ids[:self.num_reviews]):
            review = self.review_dict[review_id]
            user_features.append(self._user_feats[review['user_id']])

            review_stars[i] = int(review['stars'])
            text = review['text']
            sentences = text.split("\n")
            sentences = sentences[:self.num_sentences]

            for j, sentence in enumerate(sentences):
                tokens = self.vocab.word_splitter.split_words(sentence)
                words = [token.text for token in tokens]
                words = words[:self.max_sequence_length]
                for k, word in enumerate(words):
                    product_tensor[i, j, k] = self.vocab.word_index.get(word, self.vocab.word_index[UNKNOWN])

        user_features = torch.FloatTensor(user_features)

        product_star = torch.tensor(int((product['stars'] - 1) / 0.5), dtype=torch.long)

        output_dict = {"product": product_tensor,
                       "product_star": product_star,
                       "review_stars": review_stars,
                       "user_features": user_features}

        return output_dict

    def __len__(self):
        return len(self.products)


if __name__ == '__main__':
    vocab = Vocabulary()
    print(vocab.vocab_size)
    dataset = ProductUserDataset(vocab, './data/products.txt',
                                 './data/tokenized_reviews.txt',
                                 './data/users_feats.json')
    print(iter(dataset).__next__())
