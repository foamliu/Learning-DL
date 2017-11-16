import pandas as pd
from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import MaxEntClassifier
from textblob.classifiers import NLTKClassifier
from textblob.classifiers import DecisionTreeClassifier
import nltk

def remove_prefix(theList, prefix):
    return [text[len(prefix):] for text in theList]

if __name__ == '__main__':
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    train = list(zip(train_data.Title, remove_prefix(train_data.Team, 'UCM ')))
    test = list(zip(test_data.Title, remove_prefix(test_data.Team, 'UCM ')))

    cl = NaiveBayesClassifier(train)

    # Classify some text
    print(cl.classify("SA DRI Pls reboot monitor service"))
    print(cl.classify("Test"))
    print(cl.classify("Ticketing Job Dispatcher Reliability for Email Sender is below SLA"))
    print(cl.classify("[Support] Unknown user who pitched an opportunity on UCM-A"))
    print(cl.classify("[Support] [UCMA] For the removable tiles the colors of 'change - compared to' doesn't match positivity/negativity of change | UCM000000968815"))

    # Compute accuracy
    print("NaiveBayes Accuracy: {0}".format(cl.accuracy(test)))

    # Show 10 most informative features
    cl.show_informative_features(10)
    print(cl.informative_features(10))

    cl = DecisionTreeClassifier(train)
    print("DecisionTree Accuracy: {0}".format(cl.accuracy(test)))
    print(cl.pseudocode())

    #cl = MaxEntClassifier(train, algorithm='megam', max_iter=10, trace=0)
    #print("MaxEnt(megam) Accuracy: {0}".format(cl.accuracy(test)))

    #cl = MaxEntClassifier(train, algorithm='gis', max_iter=10, trace=0, min_lldelta=0.5)
    #print("MaxEnt(gis) Accuracy: {0}".format(cl.accuracy(test)))

    from nltk.classify.scikitlearn import SklearnClassifier
    from sklearn.naive_bayes import MultinomialNB,BernoulliNB

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(train)
    print("MultinomialNB accuracy percent:", nltk.classify.accuracy(MNB_classifier, test))

    BNB_classifier = SklearnClassifier(BernoulliNB())
    BNB_classifier.train(train)
    print("BernoulliNB accuracy percent:", nltk.classify.accuracy(BNB_classifier, test))






