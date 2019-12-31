# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 19:33
# @Author  : uhauha2929
import json
import torch
import torch.nn.functional as F
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from torch.utils.data import Dataset

from build_vocab_embedding import Vocabulary, UNKNOWN_CHAR, PADDING_CHAR


class ProductUserDatasetELMo(Dataset):
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
        self.char_mapper = ELMoCharacterMapper()

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
                                     self.char_mapper.max_word_length],
                                    self.char_mapper.padding_character + 1,  # mapper中加了1
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
                    product_tensor[i, j, k] = torch.LongTensor(self.char_mapper.convert_word_to_char_ids(word))

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
    dataset = ProductUserDatasetELMo(vocab, './data/products_train.txt',
                                     './data/tokenized_reviews.txt',
                                     './data/users_feats.json')
    output_dict = iter(dataset).__next__()
    t = output_dict['product']