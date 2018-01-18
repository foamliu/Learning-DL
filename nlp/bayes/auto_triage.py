import pandas as pd
import numpy as np
import time
from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import DecisionTreeClassifier
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopset = set(stopwords.words('english'))
ignoreDT = True

def remove_prefix(theList, prefix):
    return [text[len(prefix):] if text.startswith(prefix) else text for text in theList ]

def remove_stopwords(theList):
    return [' '.join([word for word in text.split() if word not in stopset]) for text in theList]

if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test_s.csv')
    train_data = remove_stopwords(train.Title)
    train_target = remove_prefix(train.Team, 'UCM ')
    test_data = remove_stopwords(test.Title)
    test_target = remove_prefix(test.Team, 'UCM ')
    train = list(zip(train_data, train_target))
    test = list(zip(test_data, test_target))

    start_time = time.time()
    cl = NaiveBayesClassifier(train)
    # Compute accuracy
    print("NaiveBayes Accuracy: {0}".format(cl.accuracy(test)))

    # Show 10 most informative features
    cl.show_informative_features(10)
    print(cl.informative_features(10))
    elapsed_time = time.time() - start_time
    print(elapsed_time)

    if (not ignoreDT):
        start_time = time.time()
        cl = DecisionTreeClassifier(train)
        print("DecisionTree Accuracy: {0}".format(cl.accuracy(test)))
        print(cl.pseudocode())
        elapsed_time = time.time() - start_time
        print(elapsed_time)

    start_time = time.time()
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    class StemmedCountVectorizer(CountVectorizer):
        def build_analyzer(self):
            analyzer = super(StemmedCountVectorizer, self).build_analyzer()
            return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

    text_clf = Pipeline([('vect', stemmed_count_vect),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])

    text_clf.fit(train_data, train_target)
    predicted = text_clf.predict(test_data)
    print("MultinomialNB Accuracy: {0}".format(np.mean(predicted == test_target)))
    df = pd.DataFrame(list(zip(test_data, predicted, test_target)))
    df.to_csv('MB_list.csv', index=False)
    elapsed_time = time.time() - start_time
    print(elapsed_time)

    start_time = time.time()
    from sklearn.linear_model import SGDClassifier
    text_clf = Pipeline([('vect', stemmed_count_vect),
                          ('tfidf', TfidfTransformer()),
                          ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                alpha = 1e-3, random_state = 42)),
                        ])
    text_clf.fit(train_data, train_target)

    predicted = text_clf.predict(test_data)
    print("SGD Accuracy: {0}".format(np.mean(predicted == test_target)))
    df = pd.DataFrame(list(zip(test.Id, test_data, predicted, test_target)))
    df.to_csv('SGD_list.csv', index=False)

    elapsed_time = time.time() - start_time
    print(elapsed_time)


    from sklearn import metrics
    print(metrics.classification_report(test_target, predicted))