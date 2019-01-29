import tensorflow as tf
import numpy as np
import copy


class TextRCNN(object):
    """
    A RNN-CNN for text classification/regression.
    Uses an embedding layer, followed by a recurrent, convolutional, fully-connected (and softmax) layer.
    """

    def __init__(self, w2v_model, sequence_length, num_classes, vocab_size,
                 embedding_size, rnn_size, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="word_embeddings")
            self.W = tf.get_variable("word_embeddings", initializer=w2v_model.astype(np.float32))
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # bi-lstm layer
        with tf.name_scope('bi-lstm'):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)

            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)

            (self.output_fw, self.output_bw), states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell, cell_bw=bw_cell, inputs=self.embedded_chars, dtype=tf.float32
            )

        # context
        with tf.name_scope('context'):
            shape = [tf.shape(self.output_fw)[0], 1, tf.shape(self.output_fw)[2]]
            self.c_left = tf.concat([tf.zeros(shape), self.output_fw[:, :-1]], axis=1, name='context_left')
            self.c_right = tf.concat([self.output_bw[:, 1:], tf.zeros(shape)], axis=1, name='context_right')

        # word representation
        with tf.name_scope('word-representation'):
            self.x = tf.concat([self.c_left, self.embedded_chars, self.c_right], axis=2, name='x')
            embedding_size = 2 * rnn_size + embedding_size

        # text representation
        with tf.name_scope('text_representation'):
            W2 = tf.Variable(tf.random_uniform([embedding_size, rnn_size], -1.0, 1.0), name='W2')
            b2 = tf.Variable(tf.constant(0.1, shape=[rnn_size]), name='b2')
            self.y2 = tf.tanh(tf.einsum('aij,jk->aik', self.x, W2) + b2)

        # max pooling
        with tf.name_scope('max_pooling'):
            self.y3 = tf.reduce_max(self.y2, axis=1)

        # final scores and predictions
        with tf.name_scope('output'):
            W4 = tf.get_variable(
                "W4",
                shape=[rnn_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b4 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b4')
            l2_loss += tf.nn.l2_loss(W4)
            l2_loss += tf.nn.l2_loss(b4)
            self.scores = tf.nn.xw_plus_b(self.y3, W4, b4, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

            # Calculate mean cross-entropy loss, or root-mean-square error loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")