# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 19:33
# @Author  : uhauha2929
import json
import torch
from allennlp.data import Instance, Vocabulary, Token
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.nn.util import move_to_device
from torch.utils.data import Dataset

from config import conf_bert, DEVICE


class ProductUserDatasetBERT(Dataset):
    def __init__(self,
                 products_path: str,
                 reviews_path: str,
                 user_feats_path: str,
                 num_reviews: int = 10,
                 num_sentences: int = 20,
                 max_sequence_length: int = 30):

        self.num_reviews = num_reviews
        self.num_sentences = num_sentences
        self.max_sequence_length = max_sequence_length

        self.vocab = Vocabulary()

        self.bert_indexer = PretrainedBertIndexer(pretrained_model=conf_bert.bert_vocab)
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

    def text_to_instance(self, text):
        tokens = [Token(word) for word in self.word_tokenizer(text.lower())]  # lowercase

        field = TextField(tokens, self.token_indexers)
        field.index(self.vocab)

        fields = {'tokens': field}

        instance = Instance(fields)

        return instance

    def __getitem__(self, index):
        product_tensor = torch.zeros([self.num_reviews,
                                      self.num_sentences,
                                      conf_bert.bert_dim])  # float类型
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
            if len(sentences) == 0:
                print('aasssss')
            instances = []
            for sentence in sentences:
                if len(sentence) > 0:  # IndexError: list index out of range
                    instance = self.text_to_instance(sentence)
                    instances.append(instance)
            batch = Batch(instances).as_tensor_dict(
                padding_lengths={'tokens': {'num_tokens': self.max_sequence_length}})
            batch['tokens'] = move_to_device(batch['tokens'], DEVICE.index)  # gpu
            embeddings = self.word_embedder.forward(batch['tokens'])
            product_tensor[i, :len(instances)] = embeddings[:, 0]

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
    dataset = ProductUserDatasetBERT('./data/products.txt',
                                     './data/tokenized_reviews.txt',
                                     './data/users_feats.json')

    output_dict = iter(dataset).__next__()
    for i, output_dict in enumerate(dataset):
        t = output_dict['product']
        print(i, t.size())
