import pandas as pd
import numpy as np
import xgboost as xgb
from gensim.models import Word2Vec
import jieba

jieba.initialize()
import re
from sklearn import preprocessing
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import warnings

warnings.filterwarnings('ignore')
path = 'F:/BDCI/CarNLP/data'

def load_data():
    train = pd.read_csv(path + '/train_2.csv')
    test = pd.read_csv(path + '/test_public_2.csv')
    train['content_id'] = train['content_id'].map(str)
    test['content_id'] = test['content_id'].map(str)
    return train, test

def split_words(dataset):
    data = dataset.copy()
    jieba.add_word('森林人')
    words = list(map(lambda x: jieba.cut(''.join(re.findall(u'[\u4e00-\u9fff]+', x)), cut_all=True, HMM=True),
                     data['content']))  # jieba分词,只切中文
    #    words = list(map(lambda x : jieba.cut(x,cut_all = False,HMM = True),data['content'])) # jieba分词,只切中文
    words = [list(word) for word in words]  # 分词结果转换为list
    data['words'] = words
    return data

def tfidf_improve(train, test):
    tr = train.copy()
    te = test.copy()
    tr['l'] = 'tr'
    te['l'] = 'te'
    data = pd.concat([tr, te], axis=0)
    data.index = range(len(data))
    words = data['words'].tolist()
    words = [n for a in words for n in a]  # 转一维列表
    f = open(path + '/stopwords.txt')
    stopwords = []
    for row in f.readlines():
        stopwords.append(row)
    f.close()
    stopwords = [word.replace('\n', '').strip() for word in stopwords]
    words = [word for word in words if word not in stopwords]
    print(len(words))
    words = dict(Counter(words))
    words = {k: v for k, v in words.items() if v > 15}
    print(len(words))
    words = list(words.keys())
    print(len(words))
    data['words'] = data['words'].map(lambda x: list(set(x) & set(words)))
    data['words'] = data['words'].map(lambda x: ' '.join(x))

    # tfidf
    vectorizer = CountVectorizer(token_pattern='\w+')  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(data['words'].tolist()))
    words = vectorizer.get_feature_names()
    print(len(words))
    weight = tfidf.toarray()
    weight = pd.DataFrame(weight, columns=words)
    data = pd.concat([data, weight], axis=1)
    data.fillna(0, downcast='infer', inplace=True)
    tr = data[data['l'] == 'tr']
    te = data[data['l'] == 'te']
    tr.drop(['l'], axis=1, inplace=True)
    te.drop(['l'], axis=1, inplace=True)
    tr.index = range(len(tr))
    te.index = range(len(te))
    return tr, te

def model_xgb(train, test, label, params, rounds):
    train_y = train[label]
    train_x = train.drop(
        ['sub_价格', 'sub_内饰', 'sub_动力', 'sub_外观', 'sub_安全性', 'sub_操控', 'sub_油耗', 'sub_空间', 'sub_舒适性', 'sub_配置',
         'content_id', 'content', 'sentiment_value', 'sentiment_word', 'subject', 'words'], axis=1)
    test_x = test.drop(['content_id', 'content', 'sentiment_value', 'sentiment_word', 'subject', 'words'], axis=1)
    xgb_train = xgb.DMatrix(train_x, label=train_y)
    xgb_test = xgb.DMatrix(test_x)
    # 训练
    print('开始训练!')
    watchlist = [(xgb_train, 'train')]
    bst = xgb.train(params, xgb_train, num_boost_round=rounds, evals=watchlist)
    # 预测
    print('开始预测!')
    predict = bst.predict(xgb_test)
    predict = pd.DataFrame(predict)
    return predict

if __name__ == '__main__':
    train, test = load_data()
    train = split_words(train)
    test = split_words(test)
    train, test = tfidf_improve(train, test)

    # 二分类主题
    label_data = pd.get_dummies(train['subject'], prefix='sub')
    train = pd.concat([train, label_data], axis=1)
    res_subject = test[['content_id', 'content']]

    # 1、价格
    label_name = '价格'
    label = 'sub_' + str(label_name) + ''
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.03,
        'max_depth': 6,
        'colsample_bytree': 0.9,
        'subsample': 0.9,
        'scale_pos_weight': 1,
        'min_child_weight': 0,
        'lambda': 2,
        'gamma': 0.1
    }
    num_rounds =240  # 迭代次数
    predict = model_xgb(train, test, label, params, num_rounds)
    res_subject['sub_' + str(label_name) + ''] = predict

    # 2、内饰
    label_name = '内饰'
    label = 'sub_' + str(label_name) + ''
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.03,
        'max_depth': 6,
        'colsample_bytree': 0.9,
        'subsample': 0.9,
        'scale_pos_weight': 1,
        'min_child_weight': 0,
        'lambda': 2,
        'gamma': 0.1
    }
    num_rounds = 240  # 迭代次数
    predict = model_xgb(train, test, label, params, num_rounds)
    res_subject['sub_' + str(label_name) + ''] = predict

    # 3、动力
    label_name = '动力'
    label = 'sub_' + str(label_name) + ''
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.03,
        'max_depth': 6,
        'colsample_bytree': 0.9,
        'subsample': 0.9,
        'scale_pos_weight': 1,
        'min_child_weight': 0,
        'lambda': 2,
        'gamma': 0.1
    }
    num_rounds = 240  # 迭代次数
    predict = model_xgb(train, test, label, params, num_rounds)
    res_subject['sub_' + str(label_name) + ''] = predict

    # 4、外观
    label_name = '外观'
    label = 'sub_' + str(label_name) + ''
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.03,
        'max_depth': 6,
        'colsample_bytree': 0.9,
        'subsample': 0.9,
        'scale_pos_weight': 1,
        'min_child_weight': 0,
        'lambda': 2,
        'gamma': 0.1
    }
    num_rounds =240  # 迭代次数
    predict = model_xgb(train, test, label, params, num_rounds)
    res_subject['sub_' + str(label_name) + ''] = predict

    # 5、安全性
    label_name = '安全性'
    label = 'sub_' + str(label_name) + ''
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.03,
        'max_depth': 6,
        'colsample_bytree': 0.9,
        'subsample': 0.9,
        'scale_pos_weight': 1,
        'min_child_weight': 0,
        'lambda': 2,
        'gamma': 0.1
    }
    num_rounds = 240  # 迭代次数
    predict = model_xgb(train, test, label, params, num_rounds)
    res_subject['sub_' + str(label_name) + ''] = predict

    # 6、 操控
    label_name = '操控'
    label = 'sub_' + str(label_name) + ''
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.03,
        'max_depth': 6,
        'colsample_bytree': 0.9,
        'subsample': 0.9,
        'scale_pos_weight': 1,
        'min_child_weight': 0,
        'lambda': 2,
        'gamma': 0.1
    }
    num_rounds = 240 # 迭代次数
    predict = model_xgb(train, test, label, params, num_rounds)
    res_subject['sub_' + str(label_name) + ''] = predict

    # /7、油耗
    label_name = '油耗'
    label = 'sub_' + str(label_name) + ''
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.03,
        'max_depth': 6,
        'colsample_bytree': 0.9,
        'subsample': 0.9,
        'scale_pos_weight': 1,
        'min_child_weight': 0,
        'lambda': 2,
        'gamma': 0.1
    }
    num_rounds =240  # 迭代次数
    predict = model_xgb(train, test, label, params, num_rounds)
    res_subject['sub_' + str(label_name) + ''] = predict

    # /8、空间
    label_name = '空间'
    label = 'sub_' + str(label_name) + ''
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.03,
        'max_depth': 6,
        'colsample_bytree': 0.9,
        'subsample': 0.9,
        'scale_pos_weight': 1,
        'min_child_weight': 0,
        'lambda': 2,
        'gamma': 0.1
    }
    num_rounds = 240  # 迭代次数
    predict = model_xgb(train, test, label, params, num_rounds)
    res_subject['sub_' + str(label_name) + ''] = predict

    # /9、舒适性
    label_name = '舒适性'
    label = 'sub_' + str(label_name) + ''
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.03,
        'max_depth': 6,
        'colsample_bytree': 0.9,
        'subsample': 0.9,
        'scale_pos_weight': 1,
        'min_child_weight': 0,
        'lambda': 2,
        'gamma': 0.1
    }
    num_rounds = 240  # 迭代次数
    predict = model_xgb(train, test, label, params, num_rounds)
    res_subject['sub_' + str(label_name) + ''] = predict

    # 10、配置
    label_name = '配置'
    label = 'sub_' + str(label_name) + ''
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.03,
        'max_depth': 6,
        'colsample_bytree': 0.9,
        'subsample': 0.9,
        'scale_pos_weight': 1,
        'min_child_weight': 0,
        'lambda': 2,
        'gamma': 0.1
    }
    num_rounds = 240  # 迭代次数
    predict = model_xgb(train, test, label, params, num_rounds)
    res_subject['sub_' + str(label_name) + ''] = predict

    res_subject.set_index(['content_id', 'content'], inplace=True)
    test_xy = res_subject.stack()
    test_xy = pd.DataFrame(test_xy)
    test_xy.reset_index(inplace=True)
    test_xy['subject'] = test_xy['level_2']
    test_xy.drop(['level_2'], axis=1, inplace=True)
    test_xy.sort_values(0, ascending=False, inplace=True)
    test_xy.drop_duplicates(['content_id', 'content', 'subject'], keep='first', inplace=True)
    #
    res_subject = test_xy.drop_duplicates('content_id', keep='first')
    res_subject = pd.concat([res_subject, test_xy], axis=0)
    res_subject.drop_duplicates(['content_id', 'content', 'subject'], keep='first', inplace=True)
    res_subject = res_subject.head(round(len(train) / len(set(train['content_id'])) * len(set(test['content_id']))))

    res_subject = res_subject[['content_id', 'content', 'subject']]
    res_subject['subject']=res_subject['subject'].map(lambda x:x.split('_')[1])


#    # sentiment_value
    res_sentiment_value = test[['content_id', 'content']]
    train['sen_1'] = [1 if i == 1 else 0 for i in train['sentiment_value']]
    train['sen_0'] = [1 if i == 0 else 0 for i in train['sentiment_value']]
    train['sen_-1'] = [1 if i == -1 else 0 for i in train['sentiment_value']]
    for i in range(-1, 2):
        train_x = train.drop(
        ['sen_1','sen_0','sen_-1','sub_价格', 'sub_内饰', 'sub_动力', 'sub_外观', 'sub_安全性', 'sub_操控', 'sub_油耗', 'sub_空间', 'sub_舒适性', 'sub_配置',
         'content_id', 'content', 'sentiment_value', 'sentiment_word', 'subject', 'words'], axis=1)
        test_x = test.drop(['content_id', 'content', 'sentiment_value', 'sentiment_word', 'subject', 'words'], axis=1)
        train_y = train['sen_' + str(i) + '']
        xgb_train = xgb.DMatrix(train_x, label=train_y)
        xgb_test = xgb.DMatrix(test_x)
        params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'eta': 0.03,
            'max_depth': 6,
            'colsample_bytree': 0.9,
            'subsample': 0.9,
            'scale_pos_weight': 1,
            'min_child_weight': 0,
            'lambda': 2,
            'gamma': 0.1
        }
        num_rounds = 240  # 迭代次数
        watchlist = [(xgb_train, 'train')]
        # training model
        model = xgb.train(params, xgb_train, num_boost_round=num_rounds, evals=watchlist)
        # predict test
        preds = model.predict(xgb_test)
        test_pre_y = pd.DataFrame(preds)
        res_sentiment_value['pro_sen_' + str(i) + ''] = test_pre_y

    # 归一化
    # maxs_11 = max(res_sentiment_value['pro_sen_-1'].tolist())
    # mins_11 = min(res_sentiment_value['pro_sen_-1'].tolist())
    # res_sentiment_value['pro_sen_-1'] = res_sentiment_value['pro_sen_-1'].map(lambda x: (x - mins_11) / (maxs_11 - mins_11))
    #
    # maxs_0 = max(res_sentiment_value['pro_sen_0'].tolist())
    # mins_0 = min(res_sentiment_value['pro_sen_0'].tolist())
    # res_sentiment_value['pro_sen_0'] = res_sentiment_value['pro_sen_0'].map(lambda x: (x - mins_0) / (maxs_0 - mins_0))
    #
    # maxs_1 = max(res_sentiment_value['pro_sen_1'].tolist())
    # mins_1 = min(res_sentiment_value['pro_sen_1'].tolist())
    # res_sentiment_value['pro_sen_1'] = res_sentiment_value['pro_sen_1'].map(lambda x: (x - mins_1) / (maxs_1 - mins_1))

    res_sentiment_value['pro_s'] = list(
        map(lambda x, y, z: [x, y, z], res_sentiment_value['pro_sen_-1'], res_sentiment_value['pro_sen_0'],
            res_sentiment_value['pro_sen_1']))
    res_sentiment_value['sentiment_value'] = res_sentiment_value['pro_s'].map(lambda x: x.index(max(x)) - 1)
    res_sentiment_value.drop(['pro_sen_-1', 'pro_sen_0', 'pro_sen_1', 'pro_s'], axis=1)


    # 合并
    result = pd.merge(res_subject, res_sentiment_value, on=['content_id', 'content'], how='left')
    result['sentiment_word'] = np.nan
    result['sentiment_value'] = result['sentiment_value'].map(int)

    result = result[['content_id', 'subject', 'sentiment_value', 'sentiment_word']]
    #    result = result.sample(frac = 1)
    result.to_csv('f:/liushen1.csv', index=False)







