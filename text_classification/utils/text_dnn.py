import tensorflow as tf
import numpy as np


class TextDNN(object):
    """
    A deep neural network for text classification/regression.
    Uses an embedding layer, followed by several fully-connected (and softmax) layer.
    """
    def __init__(
      self, w2v_model, sequence_length, num_classes, vocab_size,
      embedding_size, hidden_layers, hidden_size, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.get_variable("word_embeddings", initializer=w2v_model.astype(np.float32))
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # concatenate the [None, sequence_length * embedding_size] as features
            feature_size = sequence_length * embedding_size
            x = tf.reshape(self.embedded_chars_expanded, [-1, feature_size])

        # Create fully-connected layers
            with tf.name_scope("fully-connected"):
                def fc(x, num_hidden_units, name, dtype=tf.float32):
                    with tf.variable_scope(name):
                        in_dim = x.get_shape().as_list()[-1]
                        d = 1.0 / np.sqrt(in_dim)
                        w = tf.get_variable('W', shape=[in_dim, num_hidden_units], dtype=dtype,
                                            initializer=tf.random_uniform_initializer(-d, d))
                        b = tf.get_variable('b', shape=[num_hidden_units], dtype=dtype,
                                            initializer=tf.random_uniform_initializer(-d, d))
                        output = tf.matmul(x, w) + b
                        return output

            for i in range(hidden_layers):
                x = tf.nn.elu(fc(x, hidden_size, "l{}".format(i + 1)))
            self.output = x

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.output, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[hidden_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.pre = tf.nn.softmax(tf.matmul(self.h_drop, W) + b, name="pre")


        # Calculate mean cross-entropy loss, or root-mean-square error loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")