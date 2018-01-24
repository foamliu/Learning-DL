import tensorflow as tf
import numpy as np
import pandas as pd
import string
import re
import collections
import itertools

from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

stop = set(stopwords.words('english'))


def tokenize(text):
    try:
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        text = regex.sub(" ", text)  # remove punctuation

        tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent
        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
        filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
        filtered_tokens = [w.lower() for w in filtered_tokens if len(w) >= 3]

        return filtered_tokens

    except TypeError as e:
        print(text, e)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.shape)
print(test.shape)

train = train.loc[:, ['Title', 'Team']]
test = test.loc[:, ['Title', 'Team']]

train['Token'] = train['Title'].map(tokenize)
test['Token'] = test['Title'].map(tokenize)

title_tokens = np.append(train['Token'].values, test['Token'].values)
tokens = set(itertools.chain.from_iterable(title_tokens)) | set(['PAD'])

token_num_map = dict(zip(tokens, range(len(tokens))))
token_to_num = lambda token: token_num_map.get(token, len(tokens))

features = [list(map(token_to_num, tokens)) for tokens in title_tokens]

time_steps = max([len(item) for item in features])

PAD_number = token_to_num('PAD')
for vector in features:
    while (len(vector) < time_steps):
        vector.append(PAD_number)

all_teams = np.append(train['Team'].values, test['Team'].values)
teams = set([team for team in all_teams])
team_num_map = dict(zip(teams, range(len(teams))))
team_to_num = lambda team: team_num_map.get(team, len(teams))

team_vector = [team_to_num(team) for team in all_teams]
labels = np.array(team_vector).reshape(len(team_vector), -1)
enc = OneHotEncoder()
enc.fit(labels)
labels = enc.transform(labels).toarray().tolist()

feature_labels = list(zip(features, labels))

train_feature_labels = np.array(feature_labels[:train.shape[0]])
test_feature_labels = np.array(feature_labels[train.shape[0] + 1:])


class DataSet:
    def __init__(self, feature_labels, batch_size):
        indices = np.random.permutation(len(feature_labels))

        self.feature_labels = feature_labels[indices]
        self.start_index = 0
        self.batch_size = batch_size
        self.max_length = len(feature_labels)

    def next_batch(self):
        feature = [row[0] for row in self.feature_labels][
                  self.start_index: min(self.start_index + self.batch_size, self.max_length)]
        label = [row[1] for row in self.feature_labels][
                self.start_index: min(self.start_index + self.batch_size, self.max_length)]
        self.start_index += self.batch_size

        return np.array(feature), np.array(label)


batch_size = 4
num_units = 512
learning_rate = 0.00001
n_classes = len(labels[0])
n_layers = 4

x = tf.placeholder(tf.int32, [batch_size, time_steps])
y = tf.placeholder(tf.int32, [batch_size, n_classes])

out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))
embedding = tf.Variable(tf.random_normal([len(tokens), num_units]))
inputs = tf.nn.embedding_lookup(embedding, x)
inputs = tf.unstack(inputs, time_steps, 1)

cell = tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True)
lstm_layer = tf.contrib.rnn.MultiRNNCell([cell] * n_layers, state_is_tuple=True)

outputs, last_state = tf.nn.static_rnn(lstm_layer, inputs, dtype=tf.float32)
logits = tf.matmul(outputs[-1], out_weights) + out_bias
prob = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    train_dataSet = DataSet(train_feature_labels, batch_size)
    iter_times = len(train_feature_labels) // batch_size

    for iter in range(1000):
        batch_x, batch_y = train_dataSet.next_batch()
        sess.run(opt, feed_dict={x: batch_x, y: batch_y})
        if iter % 50 == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})

            print("Iter ", iter)
            print("Accuracy ", acc)
            print("Loss ", los)
            print("__________________")

    test_dataSet = DataSet(test_feature_labels, batch_size)
    iter_times = len(test_feature_labels) // batch_size

    reduce_accuracy = 0
    for iter in range(iter_times):
        test_batch_x, test_batch_y = test_dataSet.next_batch()
        acc = sess.run(accuracy, feed_dict={x: test_batch_x, y: test_batch_y})
        reduce_accuracy += acc
        print("Test Accuracy:", acc)

    print("Overall Accuracy", reduce_accuracy / (iter_times))