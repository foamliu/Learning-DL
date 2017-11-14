import pandas as pd
from textblob.classifiers import NaiveBayesClassifier

if __name__ == '__main__':
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    train = list(zip(train_data.Title, train_data.Team))
    test = list(zip(test_data.Title, test_data.Team))
    cl = NaiveBayesClassifier(train)
    print(cl.classify("SA DRI Pls reboot monitor service"))
    print(cl.classify("Test"))
    print(cl.classify("Ticketing Job Dispatcher Reliability for Email Sender is below SLA"))
    print(cl.classify("[Support] Unknown user who pitched an opportunity on UCM-A"))
    print(cl.classify("[Support] [UCMA] For the removable tiles the colors of 'change - compared to' doesn't match positivity/negativity of change | UCM000000968815"))
    print(cl.accuracy(test))
