#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 18:13:56 2017

@author: yangliu
"""

from __future__ import print_function
import collections
import numpy as np
import random
import time
import os
from os import walk
from os.path import join
from six.moves import range
import tensorflow as tf
from tensorflow.contrib import rnn
import jieba


def cleanse(content):
    content = content.replace('\n','')
    content = content.replace('\r','')
    content = content.replace('\u3000','')
    content = content.replace(' ','')
    content = content.replace('\t','')
    content = content.replace('，','')
    content = content.replace('。','')
    content = content.replace('”','')
    content = content.replace('“','')
    content = content.replace('？','')
    content = content.replace('\'','')
    content = content.replace('（','')
    content = content.replace('）','')
    content = content.replace('【','')
    content = content.replace('】','')
    content = content.replace('的','')
    content = content.replace('了','')
    
    return content


def load_file(folder):
    paths = [join(dirpath, name)
        for dirpath, dirs, files in walk(folder)
        for name in files
        if not name.startswith('.') ]
    concat = u''
    for path in paths:
        with open(path, 'r', encoding='utf-8') as myfile:
            #print(path)
            content = myfile.read()
            concat += cleanse(content)

    return concat


def read_data(concat):
    """Extract as a list of words"""
    seg_list = jieba.cut(concat, cut_all=False)
    #print "/".join(seg_list)
    words = []
    for t in seg_list:
        words.append(t)

    return words


def build_dataset(words):
    #vocab = set(words)
    #vocab_size = len(vocab)
    print('Vocabulary size %d' % vocab_size)

    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
            data.append(index)
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        #data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


#def load_embeddings():
#    import pandas as pd
#    df = pd.read_csv("final_embeddings.csv")
#    embeddings = df.as_matrix()[:,1:].astype(np.float32)
#    print('embeddings.shape: ' + str(embeddings.shape))
#    print('embeddings.dtype: ' + str(embeddings.dtype))
#    return embeddings
#
#def generate_batch():
#    global offset
#    end_offset = n_input + 1
#
#    batch = np.zeros([batch_size, n_input, embedding_size])
#    labels = np.zeros([batch_size, embedding_size])
#    for batch_index in range(batch_size):
#        # Generate a minibatch. Add some randomness on selection process.
#        if offset > (len(data)-end_offset):
#            offset = random.randint(0, n_input+1)
#        for i in range(0, n_input):
#            batch[batch_index, i] = embeddings[data[offset + i]]
#        labels[batch_index] = embeddings[data[offset + n_input]]
#
#        offset += (n_input+1)
#
#    batch = batch.reshape((-1, n_input * embedding_size)).astype(np.float32)
#    labels = labels.reshape((-1, embedding_size)).astype(np.float32)
#    #print('batch.shape: ' + str(batch.shape)) -> (128, 384)
#    return batch, labels
#
#
#def nearest(embed):
#    similarity = tf.matmul(embeddings, embed, transpose_b=True)
#    return tf.argmax(similarity)
    
def generate_batch():
    global offset
    end_offset = n_input + 1
    
    batch = np.zeros([batch_size, n_input, 1])
    labels = np.zeros([batch_size, vocab_size])
    for batch_index in range(batch_size):
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(data)-end_offset):
            offset = random.randint(0, n_input+1)
        for i in range(0, n_input):
            batch[batch_index, i] = data[offset + i]
        #labels[batch_index] = data[offset + n_input]
        labels[batch_index] = np.zeros([vocab_size], dtype=float)
        labels[batch_index][data[offset+n_input]] = 1.0
        
        offset += (n_input+1)
    
    batch = batch.reshape((-1, n_input, 1)).astype(np.float32)
    labels = labels.reshape((-1, vocab_size)).astype(np.float32)
    return batch, labels


def tf_lstm():

    # Parameters
    learning_rate = 0.0001
    training_iters = 50000
    display_step = 1000
    n_input = 3
    
    # number of units in RNN cell
    n_hidden = 1024
    num_classes = vocab_size

    graph = tf.Graph()
    with graph.as_default():
        # tf Graph input
        x = tf.placeholder("float", [None, n_input, 1])
        y = tf.placeholder("float", [None, num_classes])
        # RNN output node weights and biases
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, num_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([num_classes]))
        }
        def RNN(x, weights, biases):
            # reshape to [1, n_input]
            x = tf.reshape(x, [-1, n_input])
            # Generate a n_input-element sequence of inputs
            # (eg. [had] [a] [general] -> [20] [6] [33])
            x = tf.split(x,n_input,1)
            # 2-layer LSTM, each layer has n_hidden units.
            # Average Accuracy= 95.20% at 50k iter
            
            rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
            # 1-layer LSTM with n_hidden units but with lower accuracy.
            # Average Accuracy= 90.60% 50k iter
            # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
            # rnn_cell = rnn.BasicLSTMCell(n_hidden)
            # generate prediction
            outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
            # there are n_input outputs but
            # we only want the last output
            #print(outputs)
            return tf.matmul(outputs[-1], weights['out']) + biases['out']
        
        pred = RNN(x, weights, biases)
        
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
        # Model evaluation
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        # Create a summary to monitor cost tensor
        tf.summary.scalar("cost", cost)
        # Create a summary to monitor accuracy tensor
        tf.summary.scalar("accuracy", accuracy)

        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        
        # Initializing the variables
        init = tf.global_variables_initializer()


    # config = tf.ConfigProto(device_count = {'GPU': 0})
    # Launch the graph
    #with tf.Session(graph=graph, config=config) as session:
    with tf.Session(graph=graph) as session:
        session.run(init)
        step = 0
        offset = random.randint(0,n_input+1)

        writer.add_graph(session.graph)
        while step < training_iters:            
            batch_x, batch_y = generate_batch()
            _, acc, loss, onehot_pred, summary = session.run([optimizer, accuracy, cost, pred, merged_summary_op], \
                                                    feed_dict={x: batch_x, y: batch_y})
            if (step+1) % display_step == 0:
                print("Iter= " + str(step+1) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.2f}%".format(100*acc))

                symbols_in = [reverse_dictionary[data[i]] for i in range(offset, offset + n_input)]
                symbols_out = reverse_dictionary[data[offset + n_input]]
                #print('onehot_pred.shape： ' + str(onehot_pred.shape))
                #print('onehot_pred[-1,:].shape: ' + str(onehot_pred[-1,:].shape))
                argmax = tf.argmax(onehot_pred[-1,:]).eval()
                print('tf.argmax(onehot_pred[-1,:]).eval(): ' + str(argmax))
                symbols_out_pred = reverse_dictionary[argmax]
                print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
                writer.add_summary(summary, step)
            step += 1
        print("Optimization Finished!")
        print("Elapsed time: ", elapsed(time.time() - start_time))
        print("Run on command line.")
        print("\ttensorboard --logdir=%s" % (logdir))
        print("Point your web browser to: http://localhost:6006/")
        while True:
            prompt = "%s words: " % n_input
            sentence = input(prompt)
            sentence = sentence.strip()
            words = sentence.split(' ')
            if len(words) != n_input:
                continue
            try:
                symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
                for i in range(32):
                    keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                    onehot_pred = session.run(pred, feed_dict={x: keys})
                    onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                    sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                    symbols_in_keys = symbols_in_keys[1:]
                    symbols_in_keys.append(onehot_pred_index)
                print(sentence)
            except:
                print("Word not in dictionary")





if __name__ == '__main__':
    folder = '《刘慈欣作品全集》(v1.0)'
    concat = load_file(folder)
    print("Full text length %d" %len(concat))

    words = read_data(concat)
    print('Data size %d' % len(words))

    vocab_size = 5000
    data, count, dictionary, reverse_dictionary = build_dataset(words)
    print('Most common words ', count[:10])
    print('Sample data', data[:10])
    del words  # Hint to reduce memory.

    #embeddings = load_embeddings()

    # Target log path
    log_path = '/tmp/tensorflow/rnn_words'
    logdir = log_path
    
    previous_runs = os.listdir(logdir)
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
    
    rundir = 'run_%02d' % run_number
    logdir = os.path.join(logdir, rundir)
    writer = tf.summary.FileWriter(logdir)
    
    n_input = 3
    batch_size = 128
    #batch_size = 16
    embedding_size = 128
    offset = 0
    
    tf_lstm()
