# -*- coding: utf-8 -*-
# @Time    : 2019/5/5 10:44
# @Author  : uhauha2929
import json


def main():
    max_sent_count = -1
    min_sent_count = 10000

    max_sent_len = -1
    min_sent_len = 10000

    sent_count = []
    sent_len = []
    with open('data/old/reviews_test.txt', 'rt') as f:
        for line in f:
            r = json.loads(line)
            sents = r['text'].split('\n')
            max_sent_count = max(max_sent_count, len(sents))
            min_sent_count = min(min_sent_count, len(sents))
            sent_count.append(len(sents))
            for s in sents:
                words = s.split()
                max_sent_len = max(max_sent_len, len(words))
                min_sent_len = min(min_sent_len, len(words))
                sent_len.append(len(words))

    avg_sent_count = sum(sent_count) / len(sent_count)
    avg_sent_len = sum(sent_len) / len(sent_len)

    print(max_sent_count, min_sent_count, avg_sent_count)
    print(max_sent_len, min_sent_len, avg_sent_len)


if __name__ == '__main__':
    main()
