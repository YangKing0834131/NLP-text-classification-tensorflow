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

def tfidf(train, test):
    tr = train.copy()
    te = test.copy()
    tr['l'] = 'tr'
    te['l'] = 'te'
    data = pd.concat([tr, te], axis=0)
    data.index = range(len(data))
    data['words'] = data['words'].map(lambda x: ' '.join(x))
    # tfiddf
    vectorizer = CountVectorizer(token_pattern='\w+')  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(data['words'].tolist()))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    words = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
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
    le = preprocessing.LabelEncoder()
    le.fit(train_y)
    train_y = le.transform(train_y)
    train_x = train.drop(['content_id', 'content', 'sentiment_value', 'sentiment_word', 'subject', 'words'], axis=1)
    test_x = test.drop(['content_id', 'content', 'sentiment_value', 'sentiment_word', 'subject', 'words'], axis=1)

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)

    # 训练
    print('开始训练!')
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_boost_round=rounds, evals=watchlist)
    # 预测
    print('开始预测!')
    predict = bst.predict(dtest)
    predict = pd.DataFrame(predict)
    test_xy = test[['content_id', 'content']]
    test_xy = pd.concat([test_xy, predict], axis=1)

    for col in range(0, params['num_class']):
        test_xy.rename(columns={col: le.inverse_transform(int(col))}, inplace=True)

    test_xy.set_index(['content_id', 'content'], inplace=True)
    test_xy = test_xy.stack()
    test_xy = pd.DataFrame(test_xy)
    test_xy.reset_index(inplace=True)
    test_xy[label] = test_xy['level_2']
    test_xy.drop(['level_2'], axis=1, inplace=True)
    test_xy.sort_values(0, ascending=False, inplace=True)
    test_xy.drop_duplicates(['content_id', 'content', label], keep='first', inplace=True)
    return test_xy


if __name__ == '__main__':
    train, test = load_data()
    train = split_words(train)
    test = split_words(test)
    train, test = tfidf_improve(train, test)

    # subject
    params = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric': 'merror',
              'eta': 0.03,
              'max_depth': 6,  # 4 3
              'colsample_bytree': 0.9,  # 0.8
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'min_child_weight': 1,
              'num_class': len(set(train['subject'])),
              'lambda': 2,
              'gamma': 0.1
              }
    rounds = 240
    pre_subject = model_xgb(train, test, 'subject', params, rounds)
    res_subject = pre_subject.drop_duplicates('content_id', keep='first')
    res_subject = pd.concat([res_subject, pre_subject], axis=0)
    res_subject.drop_duplicates(['content_id', 'content', 'subject'], keep='first', inplace=True)
    res_subject = res_subject.head(round(len(train) / len(set(train['content_id'])) * len(set(test['content_id']))))

    # sentiment_value
    params = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric': 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.9,  # 0.8
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'min_child_weight': 0,
              'num_class': len(set(train['sentiment_value'])),
              'lambda': 2,
              }
    rounds =450
    pre_sentiment_value = model_xgb(train, test, 'sentiment_value', params, rounds)
    res_sentiment_value = pre_sentiment_value.drop_duplicates('content_id', keep='first')
    # 合并
    result = pd.merge(res_subject, res_sentiment_value, on=['content_id','content'], how='left')

    result['sentiment_word'] = np.nan
    result['sentiment_value'] = result['sentiment_value'].map(int)

    result = result[['content_id', 'subject', 'sentiment_value', 'sentiment_word']]
    result = result.sample(frac = 1)
    result.to_csv('f:/liushen.csv', index=False)


