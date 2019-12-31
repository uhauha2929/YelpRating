# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 19:33
# @Author  : uhauha2929
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from build_vocab_embedding import Vocabulary, UNKNOWN, PADDING


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
