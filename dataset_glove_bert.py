# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 19:33
# @Author  : uhauha2929
import json
from typing import List

import torch
from allennlp.data import Instance, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.nn.util import move_to_device
from torch.utils.data import Dataset

from allenlp_util import text_to_instance
from build_vocab_embedding import Vocabulary as my_vocab, UNKNOWN, PADDING
from config import conf_bert, DEVICE


class ProductUserDatasetGloVeBERT(Dataset):
    def __init__(self,
                 vocab: my_vocab,
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

        self.bert_indexer = PretrainedBertIndexer(pretrained_model=conf_bert.bert_vocab,
                                                  truncate_long_sequences=False)
        self.word_tokenizer = self.bert_indexer.wordpiece_tokenizer
        self.token_indexers = {'tokens': self.bert_indexer}

        self.bert_embedder = PretrainedBertEmbedder(
            pretrained_model=conf_bert.bert_dir,
            top_layer_only=True,
        ).to(DEVICE)  # gpu

        self.word_embedder: TextFieldEmbedder = \
            BasicTextFieldEmbedder({"tokens": self.bert_embedder},
                                   allow_unmatched_keys=True)

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

        instances: List[Instance] = []
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

            instance: Instance = text_to_instance(text,
                                                  self.word_tokenizer,
                                                  self.token_indexers)
            instance.index_fields(Vocabulary())
            instances.append(instance)

        batch = Batch(instances).as_tensor_dict()
        batch['tokens'] = move_to_device(batch['tokens'], DEVICE.index)  # gpu
        embeddings = self.word_embedder.forward(batch['tokens'])
        product_tensor_bert = embeddings[:, 0]

        user_features = torch.FloatTensor(user_features)

        product_star = torch.tensor(int((product['stars'] - 1) / 0.5), dtype=torch.long)

        output_dict = {"product": product_tensor,
                       "product_bert": product_tensor_bert,
                       "product_star": product_star,
                       "review_stars": review_stars,
                       "user_features": user_features}

        return output_dict

    def __len__(self):
        return len(self.products)


if __name__ == '__main__':
    vocab = my_vocab()
    print(vocab.vocab_size)
    dataset = ProductUserDatasetGloVeBERT(vocab, './data/products_train.txt',
                                          './data/tokenized_reviews.txt',
                                          './data/users_feats.json')

    # t = iter(dataset).__next__()
    for i, t in enumerate(dataset):
        print(i)
        print(t['product'].size())
        print(t['product_bert'].size())
