# -*- coding: utf-8 -*-
# @Time    : 2019/12/26 13:37
# @Author  : uhauha2929
from allennlp.data.tokenizers import Token
from allennlp.data import Instance, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder

bert_vocab = '/home/yzhao/data/bert/bert-base-uncased/vocab.txt'
bert = "/home/yzhao/data/bert/bert-base-uncased/bert-base-uncased.tar.gz"

bert_indexer = PretrainedBertIndexer(pretrained_model=bert_vocab)
token_indexers = {'tokens': bert_indexer}

bert_embedder = PretrainedBertEmbedder(
    pretrained_model=bert,
    top_layer_only=True,
)

word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                                          allow_unmatched_keys=True)

word_tokenizer = bert_indexer.wordpiece_tokenizer

vocab = Vocabulary()


def text_to_instance(text):

    tokens = [Token(word) for word in word_tokenizer(text.lower())]
    print(tokens)

    field = TextField(tokens, token_indexers)
    field.index(vocab)

    fields = {'tokens': field}

    instance = Instance(fields)
    return instance


instances = [text_to_instance('Nice to meet you!'), text_to_instance('Hello World!')]

print(instances[0].fields['tokens'])

batch = Batch(instances).as_tensor_dict()

print(batch)

embeddings = word_embedder.forward(batch['tokens'])
print(embeddings.size())
print(embeddings[:, 0])
