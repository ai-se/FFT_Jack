import time, math
import collections
import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from random import randint, random, shuffle

from ABCD import ABCD
from new_fft import FFT
from helpers import get_score

seed = 666

"split data according to target label"


def split_two(corpus, label, target_label):
    pos = []
    neg = []
    for i, lab in enumerate(label):
        if lab == target_label:
            pos.append(i)
        else:
            neg.append(i)
    positive = corpus[pos] if not isinstance(corpus, pd.DataFrame) else corpus.iloc[pos]
    negative = corpus[neg] if not isinstance(corpus, pd.DataFrame) else corpus.iloc[neg]
    # positive = [corpus[i] for i in pos]
    # negative = [corpus[i] for i in neg]
    return {'pos': positive, 'neg': negative}


"smote"


def smote(data, num, k=5):
    corpus = []
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(data)
    distances, indices = nbrs.kneighbors(data)
    for i in range(0, num):
        mid = randint(0, len(data) - 1)
        nn = indices[mid, randint(1, k)]
        datamade = []
        for j in range(0, len(data[mid])):
            gap = random()
            datamade.append((data[nn, j] - data[mid, j]) * gap + data[mid, j])
        corpus.append(datamade)
    corpus = np.array(corpus)
    return corpus


"do-FFT"

def do_FFT(train_data, test_data, train_label, test_label, clf=''):
    train_label = pd.DataFrame(train_label, columns=['Target'])
    train_label['Target'] = train_label['Target'].apply(lambda x: 1 if x == 'pos' else 0)
    train_label.set_index(train_data.index, inplace=True)
    train = pd.concat([train_data, train_label], axis=1)
    test_label = pd.DataFrame(test_label, columns=['Target'])
    test_label['Target'] = test_label['Target'].apply(lambda x: 1 if x == 'pos' else 0)
    test_label.set_index(test_data.index, inplace=True)
    test = pd.concat([test_data, test_label], axis=1)
    fft = FFT(5)
    fft.print_enabled = True
    fft.criteria = clf.split('-')[1]
    # fft.data_name = name
    fft.target = train_label.columns.values[0]
    fft.train, fft.test = train, test
    fft.build_trees()               # build and get performance on TEST data
    t_id = fft.find_best_tree()     # find the best tree on TRAIN data
    fft.eval_trees()                # eval all the trees on TEST data
    # best_structure = fft.structures[fft.best]
    TP, FP, TN, FN, pre, rec, spec, fpr, npv, acc, f1 = fft.performance_on_test[t_id]
    dist2heaven = (1-rec)**2 + (1-spec)**2
    dist2heaven = math.sqrt(dist2heaven) / math.sqrt(2)
    return pre, rec, acc, f1, dist2heaven


"sk-learn"


def do_classification(train_data, test_data, train_label, test_label, clf=''):
    if clf == 'RF':
        clf = RandomForestClassifier(random_state=seed)
    elif clf == 'LR':
        clf = linear_model.LogisticRegression(random_state=seed)
    elif clf == 'kNN':
        clf = KNeighborsClassifier()
    else:   # Decision Tree
        clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_label)
    prediction = clf.predict(test_data)
    abcd = ABCD(before=test_label, after=prediction)
    D2H = np.array([k.stats()[-1] for k in abcd()])
    F = np.array([k.stats()[4] for k in abcd()])
    A = np.array([k.stats()[3] for k in abcd()])
    P = np.array([k.stats()[1] for k in abcd()])
    R = np.array([k.stats()[0] for k in abcd()])
    labeltwo = list(set(test_label))
    if 'pos' in labeltwo[0]:
        labelone = 0
    else:
        labelone = 1
    try:
        return P[labelone], R[labelone], A[labelone], F[labelone], D2H[labelone]
    except:
        pass


"cross validation"


def cross_val(clf='', data=[], label=[], target_label='', folds=10, title=''):
    "split for cross validation"

    def cross_split(corpus, folds, index):
        i_major = []
        i_minor = []
        l = len(corpus)
        for i in range(0, folds):
            if i == index:
                i_minor.extend(range(int(i * l / folds), int((i + 1) * l / folds)))
            else:
                i_major.extend(range(int(i * l / folds), int((i + 1) * l / folds)))
        if isinstance(corpus, pd.DataFrame):
            return corpus.iloc[i_minor], corpus.iloc[i_major]
        else:
            return corpus[i_minor], corpus[i_major]

    "generate training set and testing set"

    def train_test(pos, neg, folds, index, issmote="no", neighbors=5):
        pos_test, pos_train = cross_split(pos, folds=folds, index=index)
        neg_test, neg_train= cross_split(neg, folds=folds, index=index)
        if issmote == "smote":
            num = int((len(pos_train) + len(neg_train)) / 2)
            pos_train = smote(pos_train, num, k=neighbors)
            neg_train = neg_train[np.random.choice(len(neg_train), num, replace=False)]
        if isinstance(pos_train, pd.DataFrame):
            data_train = pd.concat([pos_train, neg_train], axis=0)
            data_test = pd.concat([pos_test, neg_test], axis=0)
        else:
            data_train = np.vstack((pos_train, neg_train))
            data_test = np.vstack((pos_test, neg_test))
        label_train = ['pos'] * len(pos_train) + ['neg'] * len(neg_train)
        label_test = ['pos'] * len(pos_test) + ['neg'] * len(neg_test)

        "Shuffle"
        tmp = range(0, len(label_train))
        shuffle(tmp)
        data_train = data_train[tmp] if not isinstance(data_train, pd.DataFrame) else data_train.iloc[tmp]
        label_train = np.array(label_train)[tmp]

        tmp = range(0, len(label_test))
        shuffle(tmp)
        data_test = data_test[tmp] if not isinstance(data_test, pd.DataFrame) else data_test.iloc[tmp]
        label_test = np.array(label_test)[tmp]

        return data_train, data_test, label_train, label_test

    # data, label = make_feature(corpus, method=feature, n_features=n_feature)
    if not isinstance(data, pd.DataFrame):
        data = np.array(data)
    if not isinstance(label, pd.DataFrame):
        label = np.array(label)
    split = split_two(corpus=data, label=label, target_label=target_label)
    pos = split['pos']
    neg = split['neg']

    print(str(len(pos)) + " positive-->" + str(target_label) + " in " + str(len(label)))

    start_time = time.time()
    measures = collections.defaultdict(list)
    for i in range(folds):
        tmp = range(0, len(pos))
        shuffle(tmp)
        pos = pos[tmp] if not isinstance(pos, pd.DataFrame) else pos.iloc[tmp]
        tmp = range(0, len(neg))
        shuffle(tmp)
        neg = neg[tmp] if not isinstance(neg, pd.DataFrame) else neg.iloc[tmp]
        for index in range(folds):
            data_train, data_test, label_train, label_test = train_test(pos, neg, folds=folds, index=index)
            if clf.startswith("FFT"):
                p, r, a, f, d2h = do_FFT(data_train, data_test, label_train, label_test, clf=clf)
            else:
                p, r, a, f, d2h = do_classification(data_train, data_test, label_train, label_test, clf=clf)
            measures['precision'].append(p)
            measures['recall'].append(r)
            measures['accuracy'].append(a)
            measures['f1'].append(f)
            measures['dist2heaven'].append(d2h)
    res = measures
    print("\nTotal Runtime for [%s] in a %s-way cross val: --- %s seconds ---\n" % (title, str(folds), time.time() - start_time))
    return res


"get data & label from dict"


def get_data_label(input_dict={}, label_key=''):
    data, label = [], []
    for k, v in input_dict.iteritems():
        if k == label_key:
            label = v
        else:
            data.append[v]
    return data, label


def get_data_from_csv(input_csv):
    data, label = [], []
    jump_header = True
    with open(input_csv, 'r') as f:
        for doc in f.readlines():
            if jump_header:
                jump_header = False
                continue
            tmp = [float(i) for i in doc.split(',')]
            data.append(tmp[0:-1])
            label.append(int(tmp[-1]))
    return data, label
