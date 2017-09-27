#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:15:23 2017

@author: yangliu
"""
from __future__ import print_function

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

def reformat(dataset, labels, image_size, num_labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

def load_reformat_not_mnist(image_size, num_labels):
    pickle_file = 'notMNIST.pickle'
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
        train_dataset, train_labels = reformat(train_dataset, train_labels, image_size, num_labels)
        valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, image_size, num_labels)
        test_dataset, test_labels = reformat(test_dataset, test_labels, image_size, num_labels)
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

def tf_deep_nn(regular=False, drop_out=False, lrd=False, layer_cnt=2):
    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        hidden_node_count = 1024
        # start weight
        hidden_stddev = np.sqrt(2.0 / 784)
        weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_node_count], stddev=hidden_stddev))
        biases1 = tf.Variable(tf.zeros([hidden_node_count]))
        # middle weight
        weights = []
        biases = []
        hidden_cur_cnt = hidden_node_count
        for i in range(layer_cnt - 2):
            if hidden_cur_cnt > 2:
                hidden_next_cnt = int(hidden_cur_cnt / 2)
            else:
                hidden_next_cnt = 2
            hidden_stddev = np.sqrt(2.0 / hidden_cur_cnt)
            weights.append(tf.Variable(tf.truncated_normal([hidden_cur_cnt, hidden_next_cnt], stddev=hidden_stddev)))
            biases.append(tf.Variable(tf.zeros([hidden_next_cnt])))
            hidden_cur_cnt = hidden_next_cnt
        # first wx + b
        y0 = tf.matmul(tf_train_dataset, weights1) + biases1
        # first relu
        hidden = tf.nn.relu(y0)
        hidden_drop = hidden
        # first DropOut
        keep_prob = 0.5
        if drop_out:
            hidden_drop = tf.nn.dropout(hidden, keep_prob)
        # first wx+b for valid
        valid_y0 = tf.matmul(tf_valid_dataset, weights1) + biases1
        valid_hidden = tf.nn.relu(valid_y0)
        # first wx+b for test
        test_y0 = tf.matmul(tf_test_dataset, weights1) + biases1
        test_hidden = tf.nn.relu(test_y0)

        # middle layer
        for i in range(layer_cnt - 2):
            y1 = tf.matmul(hidden_drop, weights[i]) + biases[i]
            hidden_drop = tf.nn.relu(y1)
            if drop_out:
                keep_prob += 0.5 * i / (layer_cnt + 1)
                hidden_drop = tf.nn.dropout(hidden_drop, keep_prob)

            y0 = tf.matmul(hidden, weights[i]) + biases[i]
            hidden = tf.nn.relu(y0)

            valid_y0 = tf.matmul(valid_hidden, weights[i]) + biases[i]
            valid_hidden = tf.nn.relu(valid_y0)

            test_y0 = tf.matmul(test_hidden, weights[i]) + biases[i]
            test_hidden = tf.nn.relu(test_y0)

        # last weight
        weights2 = tf.Variable(tf.truncated_normal([hidden_cur_cnt, num_labels], stddev=hidden_stddev / 2))
        biases2 = tf.Variable(tf.zeros([num_labels]))
        # last wx + b
        logits = tf.matmul(hidden_drop, weights2) + biases2

        # predicts
        logits_predict = tf.matmul(hidden, weights2) + biases2
        valid_predict = tf.matmul(valid_hidden, weights2) + biases2
        test_predict = tf.matmul(test_hidden, weights2) + biases2

        l2_loss = 0
        # enable regularization
        if regular:
            l2_loss = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)
            for i in range(len(weights)):
                l2_loss += tf.nn.l2_loss(weights[i])
                # l2_loss += tf.nn.l2_loss(biases[i])
            beta = 0.25 / batch_size
            beta = 1e-5
            l2_loss *= beta
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels)) + l2_loss

        # Optimizer.
        if lrd:
            cur_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
            starter_learning_rate = 0.4
            learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 100000, 0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=cur_step)
        else:
            optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits_predict)
        valid_prediction = tf.nn.softmax(valid_predict)
        test_prediction = tf.nn.softmax(test_predict)
        
        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", loss)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

    num_steps = 20001

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('/tmp/nonmnist_logs/full_connect', session.graph)
        
        for step in range(num_steps):
            offset_range = train_labels.shape[0] - batch_size
            offset = (step * batch_size) % offset_range
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions, summary = session.run(
                [optimizer, loss, train_prediction, merged_summary_op], feed_dict=feed_dict)
            if step % 100 == 0:
                summary_writer.add_summary(summary, step)
            if step % 500 == 0:
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

if __name__ == '__main__':
    # First reload the data we generated in 1_notmnist.ipynb.
    image_size = 28
    num_labels = 10
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
        load_reformat_not_mnist(image_size, num_labels)
        
    # tf_better_nn(regular=True)
    # tf_better_nn(offset_range=1000)
    # tf_better_nn(offset_range=1000, drop_out=True)
    # tf_better_nn(lrd=True)
    tf_deep_nn(layer_cnt=6, lrd=True, drop_out=True, regular=True)