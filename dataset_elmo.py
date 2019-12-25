# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 19:33
# @Author  : uhauha2929
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from build_vocab import Vocabulary, UNKNOWN_CHAR, PADDING_CHAR


class ProductUserDatasetChar(Dataset):
    def __init__(self,
                 vocab: Vocabulary,
                 products_path: str,
                 reviews_path: str,
                 user_feats_path: str,
                 num_reviews: int = 10,
                 num_sentences: int = 20,
                 max_sequence_length: int = 30,
                 max_word_length: int = 50,
                 max_char_size: int = 80):

        self.num_reviews = num_reviews
        self.num_sentences = num_sentences
        self.max_sequence_length = max_sequence_length
        self.max_word_length = max_word_length

        self.vocab = vocab
        self.char_index = vocab.get_char_index(max_char_size)
        self.char_size = len(self.char_index)

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
                                     self.max_sequence_length,
                                     self.max_word_length],
                                    self.char_index[PADDING_CHAR],
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
                    for l, char in enumerate(word[:self.max_word_length]):
                        product_tensor[i, j, k, l] = self.char_index \
                            .get(char, self.char_index[UNKNOWN_CHAR])  # unk

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
                                 './data/reviews_train.txt',
                                 './data/users_feats.json')
    output_dict = iter(dataset).__next__()
    print(output_dict['product'].size())
