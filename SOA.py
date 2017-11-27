import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC

from helpers import *
from new_fft import PERFORMANCE

"Do classification jobs with the given data and learner."

def do_classification(train_data, test_data, train_label, test_label, clf='', isGMM=False):
    if not clf:
        clf = LinearSVC()
    X_train = train_data.values
    y_train = train_label.values
    X_test = test_data.values
    y_test = test_label.values
    if isGMM:
        clf.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                      for i in xrange(clf.n_components)])
        clf.fit(X_train)
    else:
        clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    tp, fp, tn, fn = get_abcd(prediction, y_test)
    pre, rec, spec, fpr, npv, acc, f1 = get_performance([tp, fp, tn, fn])
    dist2heaven = get_score("Dist2Heaven", [tp, fp, tn, fn])
    return [tp, fp, tn, fn, pre, rec, spec, fpr, npv, acc, f1], [dist2heaven]


class SOA(object):

    def __init__(self, seed=666, train=None, test=None):
        self.seed = seed
        self.train = train
        self.test = test
        SL = LogisticRegression(random_state=seed)  # simple logistic
        NB = GaussianNB()  # Naive Bayes\
        EM = GaussianMixture(random_state=seed, n_components=2, covariance_type='spherical')  # Expectation Maximization
        SMO = LinearSVC(random_state=seed)  # Support Vector Machines
        self.learners = [SL, NB, EM, SMO]
        self.names = ['SL', 'NB', 'EM', 'SMO']
        self.performances = []
        self.dist2heavens = []

    "Get performance of state of the art classifiers"

    def get_performances(self):
        # Note that the split here is DATA SENSITIVE
        train_data, train_label = self.train.iloc[:, :-1], self.train.iloc[:, -1]
        test_data, test_label = self.test.iloc[:, :-1], self.test.iloc[:, -1]
        for i, clf in enumerate(self.learners):
            performance, dist2heaven = do_classification(train_data, test_data, train_label, test_label, clf, self.names[i] == "EM")
            self.performances += [performance]
            self.dist2heavens += [dist2heaven]

    def print_soa(self):
        # print "======================================="
        # print "\t----- STATE-OF-THE-ART PERFORMANCES -----"
        # print PERFORMANCE
        for i in range(len(self.names)):

            print "\t" + self.names[i] + "    \t"+ \
                  "\t".join([str(x).ljust(5, "0") for x in self.performances[i][4:] + self.dist2heavens[i]])





