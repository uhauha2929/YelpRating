# -*- coding: utf-8 -*-
# @Time    : 2019/12/26 13:37
# @Author  : uhauha2929
from allennlp.data import Instance, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import PretrainedBertIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
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

word_splitter = BertBasicWordSplitter()

vocab = Vocabulary()


def text_to_instance(text):
    tokens = word_splitter.split_words(text)
    print(tokens)

    field = TextField(tokens, token_indexers)
    field.index(vocab)

    fields = {'tokens': field}

    instance = Instance(fields)
    return instance


instances = [text_to_instance('Nice to meet you!'), text_to_instance('Hello Word!')]

print(instances[0].fields['tokens'])
print(instances[0].as_tensor_dict())

iterator = BucketIterator(batch_size=2, sorting_keys=[("tokens", "num_tokens")])
batch = iter(iterator(instances)).__next__()
print(batch)

embeddings = word_embedder.forward(batch['tokens'])
print(embeddings.size())
print(embeddings[:, 0])
