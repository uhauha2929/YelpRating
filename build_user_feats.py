# -*- coding: utf-8 -*-
# @Time    : 2019/4/19 14:25
# @Author  : uhauha2929
import json

from sklearn import preprocessing


def main():
    user_ids = set()
    with open('data/reviews_train.txt', 'rt') as r:
        for line in r:
            review = json.loads(line)
            user_ids.add(review['user_id'])
    print(len(user_ids))

    with open('data/reviews_test.txt', 'rt') as r:
        for line in r:
            review = json.loads(line)
            user_ids.add(review['user_id'])
    print(len(user_ids))

    users_feats = []
    user_ids_list = []
    with open('/home/yzhao/data/Yelp/user.json', 'rt') as u:
        for line in u:
            user = json.loads(line)
            feats = []
            if user['user_id'] in user_ids:
                feats.append(user['review_count'])
                feats.append(int(user['yelping_since'][:4]))
                feats.append(len(user['friends']))
                feats.append(user['useful'])
                feats.append(user['funny'])
                feats.append(user['cool'])
                feats.append(user['fans'])
                feats.append(len(user['elite']))
                feats.append(user['average_stars'])
                feats.append(user['compliment_hot'])
                feats.append(user['compliment_more'])
                feats.append(user['compliment_profile'])
                feats.append(user['compliment_cute'])
                feats.append(user['compliment_list'])
                feats.append(user['compliment_note'])
                feats.append(user['compliment_plain'])
                feats.append(user['compliment_cool'])
                feats.append(user['compliment_funny'])
                feats.append(user['compliment_writer'])
                feats.append(user['compliment_photos'])
                user_ids_list.append(user['user_id'])
                users_feats.append(feats)

    assert len(user_ids_list) == len(users_feats)
    print(users_feats[0])

    scaled_users_feats = preprocessing.scale(users_feats)
    print(scaled_users_feats.mean(axis=0))
    print(scaled_users_feats.std(axis=0))

    user_id_feat_dict = dict(zip(user_ids_list, scaled_users_feats.tolist()))

    with open('data/users_feats.json', 'wt') as u:
        json.dump(user_id_feat_dict, u)


if __name__ == '__main__':
    main()
