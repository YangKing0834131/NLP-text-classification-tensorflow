#! /usr/bin/env python

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.contrib import learn

from utils import data_helpers
from utils import tools

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the positive data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir","F:/CCF2018_Car_NLP&&Supply_Chain/BDCI_Train_Race/CarNLP/1024/DL/text_classification/subject_pro/runs/1547180287/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
test = pd.read_csv(tools.data_path() + 'test_public.csv')
# add=pd.DataFrame(columns =['content_id','content'])
# add['content_id']=[1,2,3,4]
# test=pd.concat([test,add])


x_text=data_helpers.cut_word(test)['words']

# Map data into vocabulary 将数据映射到词汇表
vocab_path = os.path.join(tools.vocab_dict_path(), "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_text = np.array(list(vocab_processor.transform(x_text)))

print("\nEvaluating...\n")

# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate  scores:十个类别的概率  predictions输出最大概率的类别
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        # scores = graph.get_operation_by_name("output/scores").outputs[0]


        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_text), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])


sub_prop=pd.DataFrame(all_predictions)


test = pd.read_csv(tools.data_path()+'test_public.csv')
train = pd.read_csv(tools.data_path()+'train.csv')


test_xy = test[['content_id', 'content']]
test_xy = pd.concat([test_xy, sub_prop], axis=1)
test_xy.rename(columns={ 0: 'subject'}, inplace=True)


train_y = train['subject']
le = preprocessing.LabelEncoder()
le.fit(train_y)

test_xy['subject']=test_xy['subject'].map(lambda x:le.inverse_transform(int(x)))
test_xy.to_csv(tools.result_path()+'subject.csv',index=False)


