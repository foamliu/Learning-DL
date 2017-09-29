#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:45:28 2017

@author: yangliu
"""

from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

def reformat(dataset, labels, image_size, num_labels):
    dataset = dataset.reshape((-1, image_size, image_size, 1)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
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

# convolution
def conv2d(x, W, padd):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padd)

# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def tf_conv_net(drop_out=False, lrd=False):
    batch_size = 16
    patch_size = 5
    depth = 16
    num_hidden = 64
    logdir = '/tmp/nonmnist_logs/convnet'
    
    previous_runs = os.listdir(logdir)
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
    
    rundir = 'run_%02d' % run_number
    logdir = os.path.join(logdir, rundir)

    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 1))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
      
        # Variables.
        W_conv1 = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, 1, depth], stddev=0.1))
        b_conv1 = tf.Variable(tf.zeros([depth]))
        W_conv2 = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth, depth], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(1.0, shape=[depth]))
        W_fc1 = tf.Variable(tf.truncated_normal(
          [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
        W_fc2 = tf.Variable(tf.truncated_normal(
          [num_hidden, num_labels], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(1.0, shape=[num_labels]))

        # Model.
        def model(data):
            conv = conv2d(data, W_conv1, padd='SAME')
            h_conv1 = tf.nn.relu(conv + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)
            h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
            conv = conv2d(h_norm1, W_conv2, padd='SAME')
            h_conv2 = tf.nn.relu(conv + b_conv2)
            h_norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
            h_pool2 = max_pool_2x2(h_norm2)

            shape = h_pool2.get_shape().as_list()
            reshape = tf.reshape(h_pool2, [shape[0], shape[1] * shape[2] * shape[3]])
            h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)
            h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
            return h_fc2

        # Training computation.
        logits = model(tf_train_dataset)
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
        # Optimizer.
        if lrd:
            cur_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
            starter_learning_rate = 0.04
            learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 1000, 0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=cur_step)
        else:
            optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
        test_prediction = tf.nn.softmax(model(tf_test_dataset))
        
        acc = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(tf_train_labels, 1))
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))
        
        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", loss)
        # Create a summary to monitor accuracy tensor
        tf.summary.scalar("accuracy", acc)
        # Create a summary to monitor learning_rate tensor
        tf.summary.scalar("learning_rate", learning_rate)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        
    num_steps = 20001

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logdir, session.graph)

        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions, summary = session.run(
                    [optimizer, loss, train_prediction, merged_summary_op], feed_dict=feed_dict)
            if (step % 50 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(
                        valid_prediction.eval(), valid_labels))
                summary_writer.add_summary(summary, step)
                
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

if __name__ == '__main__':
    # First reload the data we generated in 1_notmnist.ipynb.
    image_size = 28
    num_labels = 10
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
        load_reformat_not_mnist(image_size, num_labels)
 
    tf_conv_net(lrd=True, drop_out=False)