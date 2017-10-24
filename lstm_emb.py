# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:17:37 2017

@author: foamliu
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
    content = content.replace('…','')
    content = content.replace('：','')
    content = content.replace('’','')
    content = content.replace('；','')
    content = content.replace('、','')
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
            #data.append(index)
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
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


def load_embeddings():
    import pandas as pd
    embeddings = np.zeros([vocab_size, embedding_size])
    raw_emb = pd.read_csv("final_embeddings.csv", index_col=0).astype(np.float32)
    for key,value in reverse_dictionary.items():
        #print(key)
        #print(value)
        try:
            embeddings[key] = raw_emb.loc[value]
        except:
            embeddings[key] = embeddings[0]            

    return embeddings


def generate_batch(offset, batch_size, n_input):
    end_offset = n_input + 1

    batch = np.zeros([batch_size, n_input, 1], dtype=np.int32)
    labels = np.zeros([batch_size, 1], dtype=np.int32)
    for batch_index in range(batch_size):
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(data)-end_offset):
            offset = random.randint(0, n_input+1)
        for i in range(0, n_input):
            batch[batch_index, i] = data[offset + i]
        labels[batch_index] = data[offset + n_input]

        offset += (n_input+1)

    batch = batch.reshape((-1, n_input)).astype(np.float32)
    labels = labels.reshape((-1, 1)).astype(np.float32)
    #print('batch.shape: ' + str(batch.shape)) #-> (128, 384)
    return batch, labels, offset


def similarity(embed1, embed2):
    return tf.matmul(embed1, embed2, transpose_b=True)


def nearest(embed):
    similarity = tf.matmul(embeddings.as_matrix(), embed, transpose_b=True)
    return tf.argmax(similarity)


#def lookup_embedding(index):
#    return embeddings.ix[reverse_dictionary[index]]


#def lookup_embeddings(x):
#    result = np.zeros([n_input, embedding_size])
#    for i in range(0, n_input):
#        print(x[i,0])
#        result[i] = embeddings.ix[reverse_dictionary[x[i,0]]]
#    return np.reshape(result, [n_input*embedding_size])


def tf_lstm():

    # Parameters
    learning_rate = 0.0001
    training_iters = 100000
    display_step = 1000
    #save_step = 50000
    batch_size = 64
    
    # number of units in RNN cell
    n_hidden = 1024
    words_to_gen = 128

    graph = tf.Graph()
    with graph.as_default():#, tf.device('cpu:0'):
        # tf Graph input
        x = tf.placeholder(tf.int32, [None, n_input], name='x')
        y = tf.placeholder(tf.int32, [None, 1], name='y')
        # RNN output node weights and biases
        weights = tf.Variable(tf.random_normal([n_hidden, embedding_size]), name='weights')
        biases = tf.Variable(tf.random_normal([embedding_size]), name='biases')

        def RNN(x, weights, biases):
            # reshape to [1, n_input]
            x = tf.reshape(x, [-1, n_input])
            # Generate a n_input-element sequence of inputs
            # (eg. [had] [a] [general] -> [20] [6] [33])
            #x = tf.split(x, n_input,1)
            # 2-layer LSTM, each layer has n_hidden units.
            # Average Accuracy= 95.20% at 250k iter
            
            rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
            # 1-layer LSTM with n_hidden units but with lower accuracy.
            # Average Accuracy= 90.60% 250k iter
            # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
            # rnn_cell = rnn.BasicLSTMCell(n_hidden)
            # generate prediction
            outputs, states = rnn.static_rnn(rnn_cell, tf.nn.embedding_lookup(params=embeddings, ids=x), dtype=tf.float32)
            # there are n_input outputs but
            # we only want the last output
            #print(outputs)
            return tf.matmul(outputs[-1], weights) + biases
        
        pred = RNN(x, weights, biases)
        
        with tf.name_scope('cost'):
            cost = -tf.reduce_mean(similarity(pred, tf.nn.embedding_lookup(embeddings, y)))
            tf.summary.scalar(name='cost', tensor=cost)
        
        with tf.name_scope('optimizer'):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
        
        with tf.name_scope("accuracy"):
            # Model evaluation            
            correct_pred = (nearest(pred) == y)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))       
            tf.summary.scalar(name='accuracy', tensor=accuracy)
       

        summary_op = tf.summary.merge_all()
        #saver = tf.train.Saver()


    #config = tf.ConfigProto(device_count = {'GPU': 0})    
    #with tf.Session(graph=graph, config=config) as session:
    # Launch the graph
    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        step = 0
        offset = random.randint(0,n_input+1)
        writer.add_graph(session.graph)
        
        print("Run on command line.")
        print("\ttensorboard --logdir=%s" % (LOGDIR))
        print("Point your web browser to: http://localhost:6006/")
        
        iter_start = time.time()
        while step < training_iters:            
            batch_x, batch_y, offset = generate_batch(offset, batch_size, n_input)
            session.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            
            if (step+1) % display_step == 0:
                # Calculate batch accuracy
                acc, loss, emb_pred, summary = session.run([accuracy, cost, pred, summary_op], feed_dict={x: batch_x, y: batch_y})
                writer.add_summary(summary, step)
                print("Iter= " + str(step+1) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.2f}%".format(100*acc) + ", Iter time= " + \
                      elapsed(time.time() - iter_start))

                iter_start = time.time()
                symbols_in = [reverse_dictionary[data[i]] for i in range(offset - (n_input + 1), offset - 1)]
                symbols_out = reverse_dictionary[data[offset - 1]]
                #print('onehot_pred.shape： ' + str(onehot_pred.shape))
                #print('onehot_pred[-1,:].shape: ' + str(onehot_pred[-1,:].shape))
                argmax = nearest(emb_pred).eval()
                print('nearest(emb_pred).eval(): ' + str(argmax))
                symbols_out_pred = reverse_dictionary[argmax]
                print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
            
            #if (step+1) % save_step == 0:
                #print("Saving model checkpoint after {} steps.".format(step+1))
                #saver.save(session, os.path.join(MODELDIR, "model.ckpt"), (step+1))
                
            step += 1           
            
            
        print("Optimization Finished!")
        print("Elapsed time: ", elapsed(time.time() - start_time))

        
        while True:
            prompt = "%s words: " % n_input
            sentence = input(prompt)
            sentence = sentence.strip()
            words = sentence.split(' ')
            if len(words) != n_input:
                continue
            try:
                symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
                for i in range(words_to_gen):
                    keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                    emb_pred = session.run(pred, feed_dict={x: keys})
                    emb_pred_index = nearest(emb_pred).eval()
                    sentence = "%s %s" % (sentence,reverse_dictionary[emb_pred_index])
                    symbols_in_keys = symbols_in_keys[1:]
                    symbols_in_keys.append(emb_pred_index)
                print(sentence)
            except:
                print("Word not in dictionary")



if __name__ == '__main__':
    MAINDIR = os.path.dirname(os.path.realpath(__file__))
    MODELDIR = MAINDIR + "/models"
    DATADIR = MAINDIR + "/《刘慈欣作品全集》(v1.0)"
    LOGDIR = MAINDIR + "/logs"
    
    if not os.path.exists(MODELDIR):
        os.makedirs(MODELDIR)
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)

    concat = load_file(DATADIR)
    print("Full text length %d" %len(concat))

    words = read_data(concat)
    print('Data size %d' % len(words))

    vocab_size = 10000
    data, count, dictionary, reverse_dictionary = build_dataset(words)
    print('Most common words ', count[:10])
    print('Sample data', data[:10])
    del words  # Hint to reduce memory.

    # Target log path
    previous_runs = os.listdir(LOGDIR)
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
    
    rundir = 'run_%02d' % run_number
    logpath = os.path.join(LOGDIR, rundir)
    writer = tf.summary.FileWriter(logpath)
    
    embedding_size = 128
    n_input = 3
    embeddings = load_embeddings()
    
    tf_lstm()
