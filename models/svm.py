# -*- coding: utf-8 -*-
# @Time    : 2019/5/4 17:47
# @Author  : uhauha2929
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC


def main():
    stars = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    with open('../data/old/vocab.json', 'rt') as v:
        vocab = json.load(v)

    products_train = []
    with open('../data/old/products_train.txt', 'rt') as f:
        for line in f:
            products_train.append(json.loads(line))

    products_test = []
    with open('../data/old/products_test.txt', 'rt') as f:
        for line in f:
            products_test.append(json.loads(line))

    reviews_dict = {}
    with open('../data/old/tokenized_reviews.txt', 'rt') as r:
        for line in r:
            r = json.loads(line)
            reviews_dict[r['review_id']] = r['text']

    X_train = []
    y_train = []
    for p in products_train:
        text = []
        for r_id in p['review_ids']:
            text.append(reviews_dict[r_id])
        text = ' '.join(text)
        X_train.append(text)
        y_train.append(stars.index(p['stars']))

    print(len(X_train))

    X_test = []
    y_test = []
    for p in products_test:
        text = []
        for r_id in p['review_ids']:
            text.append(reviews_dict[r_id])
        text = ' '.join(text)
        X_test.append(text)
        y_test.append(stars.index(p['stars']))

    print(len(X_test))

    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    print(X_train.shape, X_test.shape)

    # clf = LogisticRegression(solver='sag', class_weight='balanced')
    clf = SVC(kernel='linear', class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))


if __name__ == '__main__':
    main()
