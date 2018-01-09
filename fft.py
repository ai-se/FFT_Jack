import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC
import collections
import pickle
import os


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

from ABCD import ABCD
SEED = 666
PRE, REC, SPEC, FPR, NPV, ACC, F1 = 7, 6, 5, 4, 3, 2, 1
COLORS = ["#800000", "#6B8E23", "#0000CD", "#FFFF00", "#8A2BE2",  "#00FF00", "#00FFFF", "#FF00FF"]
MARKERS = ['v', 2, ',', 'h',  ">", 's', '*', 'p', '8']
# training / testing split
class FFT(object):
    def __init__(self, df=None, max_level=1, goal=REC):
        cnt = 2 ** max_level
        self.train, self.test = None, None
        self.title = ""
        self.img_path = ""
        #self.train, self.test = self.split_data(df)
        self.max_level = max_level
        self.tree_cnt = cnt
        self.tree_depth = [0] * cnt
        self.target = "bug"
        self.criteria = "Dist2Heaven"
        self.goal_chase = -goal
        self.best = -1
        self.ignore = {"name", "version", 'name.1'}
        self.trees = [{}] * cnt
        self.tree_plotted = [False] * cnt
        self.selected = [{} for _ in range(cnt)]
        self.roc = [None] * self.tree_cnt
        self.tree_info = [collections.defaultdict(dict) for _ in xrange(cnt)]
        self.soa = None
        self.soa_name = ["SL", "NB", "EM", "SMO"]
        self.performance_on_train = [collections.defaultdict(dict) for _ in xrange(cnt)]
        self.performance_on_test = None


    "Get all possible tree structure"
    def get_all_structure(self):
        def dfs(cur, n):
            if len(cur) == n:
                ans.append(cur)
                return
            dfs(cur + [1], n)
            dfs(cur + [0], n)

        ans = []
        dfs([], self.max_level)
        return ans


    "Split data into training and testing"
    def split_data(self, df):
        if not df:
            return None, None
        np.random.seed(888)
        mask = np.random.rand(len(df)) <= 0.6
        train, test = df[mask], df[~mask]
        return train, test


    "Get all the metrics for the current decision"
    def get_metrics(self, data, cue, direction, threshold, decision):
        if direction == ">":
            pos, neg = data.loc[data[cue] > threshold], data.loc[data[cue] <= threshold]
        else:
            pos, neg = data.loc[data[cue] < threshold], data.loc[data[cue] >= threshold]
        if decision == 0:
            pos, neg = neg, pos
        tp = len(pos.loc[pos[self.target] == 1])
        fp = len(pos.loc[pos[self.target] == 0])
        tn = len(neg.loc[neg[self.target] == 0])
        fn = len(neg.loc[neg[self.target] == 1])
        pre, rec, spec, fpr, npv, acc, f1 = get_performance(tp, fp, tn, fn)
        return tp, fp, tn, fn, pre, rec, spec, fpr, npv, acc, f1


    "Give a cue, direction, threshold, result"
    def get_decision(self):


    "Given tree and level, get the node info on testing data"
    def get_node_info(self, t_id, i, reverse=False):
        # cue, direction, threshold, result
        c, d, t, r = self.selected[t_id][i]
        if reverse:
            d = ">" if d == "<" else "<"
        results = ["\'Good\'", "\'Bug!\'"]
        description = ("| " * i + " ".join([c, d, str(t)]) + "\t--> " + results[~r if reverse else r]).ljust(30, " ")
        self.tree_info[t_id][i]["description"] = self.tree_info[t_id][i].get("description", []) + [description]

        if "general" not in self.tree_info[t_id]:
            self.tree_info[t_id]["general"] = [0] * 5 # TP, FP, TN, FN, MCU
        tp, fp, tn, fn = self.performance_on_train[t_id][i][(c, d, t, r)][:4]
        if (r and not reverse) or (r == 0 and reverse):
            cues_used = 1.0 * (i + 1) * (tp + fp) / self.train.shape[0]
            self.tree_info[t_id]["general"] = [x + y for x, y in zip(self.tree_info[t_id]["general"], [tp, fp, 0, 0, cues_used])]
            metric = "\tFalse Alarm: " + str(fp) + ", Hit: " + str(tp)
        else:
            cues_used = 1.0 * (i + 1) * (tn + fn) / self.train.shape[0]
            self.tree_info[t_id]["general"] = [x + y for x, y in zip(self.tree_info[t_id]["general"], [0, 0, tn, fn, cues_used])]
            metric = "\tCorrect Rej: " + str(tn) + ", Miss: " + str(fn)

        self.tree_info[t_id][i]["metric"] = self.tree_info[t_id][i].get("metric", []) + [metric]


    "Get the detailed info for the specified tree"
    def describe_tree(self, t_id):
        n = self.tree_depth[t_id] + 1
        for i in range(n):
            self.get_node_info(t_id, i)
            if i == n - 1:
                self.get_node_info(t_id, i, reverse=True)


    def find_best_tree(self):
        print "#### Performance for all FFT generated. ####"
        print "-------------------------------------------------------"
        print "ID\t PRE\t REC\t SPEC\t NPV\t ACC\t F1"
        goal = self.goal_chase
        selected = None
        for i in range(self.tree_cnt):
            # self.describe_tree(i)
            # tp, fp, tn, fn, mcu = self.tree_info[i]["general"]
            # metric = get_performance(tp, fp, tn, fn)

            # tp, fp, tn, fn, mcu = self.tree_info[t_id]["general"]
            # mcu = self.tree_info[i]["general"][-1]  # note that the mcu here is for training data.
            tp, fp, tn, fn = self.performance_on_test[i][:4]    # the performance if from test data.
            metric = get_performance(tp, fp, tn, fn)

            self.roc[i] = [metric[-FPR], metric[-REC]]
            dist2heaven = self.roc[i][0]**2 + (1-self.roc[i][1])**2
            if not selected or dist2heaven < selected[1]:
                selected = [i] + [dist2heaven]
            print " \t ".join([str(x) for x in [i] + metric])
        print "-------------------------------------------------------"
        print "\nThe selected FFT id is :" + str(selected[0])
        print "The selected FFT constructed as the following tree: "

        # Get the state of the art classifiers.
        if not self.soa:
            self.get_soa()

        # plot ROC
        fig, ax = plt.subplots()
        ax.set_title('FFT splits based on: ' + self.criteria + '  |  Data: ' + self.title)
        ax.set_xlabel("False Alarm Rates")
        ax.set_ylabel("Recall")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        # plot diagonal
        x, y = [0.001 * i for i in range(1000)], [0.001 * i for i in range(1000)]
        ax.scatter(x, y, s=4)

        # plot fft peformances
        roc = self.roc
        tmp = {"Accuracy":0, "Dist2Heaven": 1, "Gini": 2, "InfoGain": 3}
        k = tmp[self.criteria]
        s_id = selected[0]
        for i in range(self.tree_cnt):
            if i == s_id:
                continue
            ax.scatter(roc[i][0], roc[i][1], c=COLORS[k], s=100)
            ax.annotate(i, (roc[i][0], roc[i][1]))
        t = 0 if s_id != 0 else 1
        ax.scatter(roc[t][0], roc[t][1], c=COLORS[k], s=100, label="FFT")

        # plot the best fft in red
        ax.scatter(roc[s_id][0], roc[s_id][1], c='r', marker=MARKERS[0], s=100, label="Best_FFT")
        ax.annotate("B_FFT", (roc[s_id][0], roc[s_id][1]))

        # plot state of the art performance
        for i in range(4):
            ax.scatter(self.soa[i][-FPR], self.soa[i][-REC], s=120,\
                       c=COLORS[i + 4], marker=MARKERS[i + 4], label=self.soa_name[i])
            ax.annotate(self.soa_name[i], (self.soa[i][-FPR], self.soa[i][-REC]))

        legend = ax.legend(loc='lower right', shadow=True, fontsize='small')
        # Put a nicer background color on the legend.
        legend.get_frame().set_facecolor('#CEE5DD')
        # plt.show()
        plt.savefig(self.img_path)
        return s_id


    def plot_tree(self, t_id=-1, show_metrics=False):
        if t_id == -1:
            t_id = self.find_best_tree()
            self.best = t_id

        if not self.tree_plotted[t_id]:
            self.tree_plotted[t_id] = True
            self.describe_tree(t_id)

        n = self.tree_depth[t_id] + 1
        for i in range(n):
            print self.tree_info[t_id][i]["description"][0] + \
                  (self.tree_info[t_id][i]["metric"][0] if show_metrics else "")
        print self.tree_info[t_id][i]["description"][1] + \
              (self.tree_info[t_id][i]["metric"][1] if show_metrics else "")

        # tp, fp, tn, fn, mcu = self.tree_info[t_id]["general"]
        mcu = self.tree_info[t_id]["general"][-1]
        tp, fp, tn, fn = self.performance_on_test[t_id][:4]
        print "\n#### Performance of the best FFT ####"
        print "======================================="
        print "\t".join([x.ljust(6, " ") for x in ["TP", "FP", "TN", "FN"]])
        print "\t".join([str(x).ljust(6, " ") for x in [tp, fp, tn, fn]])
        print "\t".join([str(x).ljust(6, " ") for x in ["PRE", "REC", "SPEC", "FPR", "NPV", "ACC", "F1"]])
        print "\t".join([str(x).ljust(6, " ") for x in [round(mcu,1)] + get_performance(tp, fp, tn, fn)])
        print "\n#### Performance of the State-Of-The-Art Models ####"
        print "======================================="
        for i in range(len(self.soa)):
            print "\t".join([str(x).ljust(6, " ") for x in [self.soa_name[i]] + self.soa[i][4:]])



    "Grow the t_id_th tree for the level with the given data"
    def grow(self, data, t_id, level):
        """
        :param data: current data for future tree growth
        :param t_id: tree id
        :param level: level id
        :return: None
        """
        if level >= self.max_level or len(data) == 0:
            return
        self.tree_depth[t_id] = level
        decision = self.structures[t_id][level]
        cur_selected = None
        for cue in list(data):
            if cue in self.ignore or cue == self.target or cue not in self.selected_feature:
                continue
            threshold = data[cue].median()
            for direction in "><":
                metrics = self.get_metrics(data, cue, direction, threshold, decision)
                self.performance_on_train[t_id][level][(cue, direction, threshold, decision)] = metrics
                goal = self.goal_chase
                if self.criteria == "Accuracy":
                    score = -metrics[-ACC]
                elif self.criteria == "Dist2Heaven":
                    score = metrics[-FPR]**2 + (1-metrics[-REC])**2
                elif self.criteria == "Gini":
                    p1 = metrics[-PRE]        # target == 1 for the positive split
                    p0 = 1 - metrics[-NPV]    # target == 1 for the negative split
                    score = 1 - p0**2 - p1**2
                else:   # information gain
                    P, N = metrics[0] + metrics[3], metrics[1] + metrics[2]
                    p = 1.0 * P / (P + N) if P + N > 0 else 0   # before the split
                    p1 = metrics[-PRE]  # the positive part of the split
                    p0 = 1 - metrics[-NPV]  # the negative part of the split
                    I, I0, I1 = (-x * np.log2(x) if x != 0 else 0 for x in (p, p0, p1))
                    I01 = p * I1 + (1-p) * I0
                    score = -(I - I01)     # the smaller the better.
                # if not cur_selected or metrics[goal] > self.performance_on_train[t_id][level][cur_selected][goal]:
                if not cur_selected or score < cur_selected[1]:
                    cur_selected = [(cue, direction, threshold, decision), score]
        self.selected[t_id][level] = cur_selected[0]
        s_cue, s_dirc, s_thre, s_decision = cur_selected[0]
        undecided = data.loc[data[s_cue] <= s_thre] if s_dirc == ">" else data.loc[data[s_cue] >= s_thre]
        self.grow(undecided, t_id, level + 1)


    "Build all possible tress."
    def build_trees(self):
        self.structures = self.get_all_structure()
        for i in range(len(self.structures)):
            self.grow(self.train, i, 0)


    "Get the performances on test data for all the FFTs."
    def get_tree_performances(self):
        if self.performance_on_test:
            return
        self.performance_on_test = [0] * len(self.structures)
        for t_id in range(len(self.structures)):
            TP, FP, TN, FN = 0, 0, 0, 0
            self.performance_on_test[t_id] = []
            level = self.tree_depth[t_id] + 1
            data = self.test
            for l_id in range(level):
                if len(data) == 0:
                    break
                cue, dirc, thre, decision = self.selected[t_id][l_id]
                tp, fp, tn, fn, pre, rec, spec, fpr, npv, acc, f1 = self.get_metrics(data, cue, dirc, thre, decision)
                TP, FP, TN, FN = TP + tp, FP + fp, TN + tn, FN + fn
                if dirc == ">":
                    left = data.loc[data[cue] <= thre]
                else:
                    left = data.loc[data[cue] >= thre]
                data = left
            pre, rec, spec, fpr, npv, acc, f1 = get_performance(tp, fp, tn, fn)
            self.performance_on_test[t_id] = [TP, FP, TN, FN, pre, rec, spec, fpr, npv, acc, f1]


    "Get performance of state of the art classifiers"
    def get_soa(self):
        train_data, train_label = self.train.iloc[:, 3:-1], self.train.iloc[:, -1]
        test_data, test_label = self.test.iloc[:, 3:-1], self.test.iloc[:, -1]

        SL = LogisticRegression(random_state=SEED)   # simple logistic
        NB = GaussianNB()           # Naive Bayes\
        EM = GaussianMixture(random_state=SEED, n_components=2, covariance_type='spherical')  # Expectation Maximization
        SMO = LinearSVC(random_state=SEED)           # Support Vector Machines
        self.soa_learners = [SL, NB, EM, SMO]
        m_SL = do_classification(train_data, test_data, train_label, test_label, SL)
        m_NB = do_classification(train_data, test_data, train_label, test_label, NB)
        m_EM = do_classification(train_data, test_data, train_label, test_label, EM, isGMM=True)
        m_SMO = do_classification(train_data, test_data, train_label, test_label, SMO)
        self.soa = [m_SL, m_NB, m_EM, m_SMO]


    def run_all(self):
        self.get_soa()
        self.build_trees()
        self.get_tree_performances()
        self.plot_tree(show_metrics=True)


"Given TP, FP, TN, FN, get all the other metrics. "
def get_performance(tp, fp, tn, fn):
    pre = 1.0 * tp / (tp + fp) if (tp + fp) != 0 else 0
    rec = 1.0 * tp / (tp + fn) if (tp + fn) != 0 else 0
    spec = 1.0 * tn / (tn + fp) if (tn + fp) != 0 else 0
    fpr = 1 - spec
    npv = 1.0 * tn / (tn + fn) if (tn + fn) != 0 else 0
    acc = 1.0 * (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) != 0 else 0
    return [round(x, 1) for x in [pre, rec, spec, fpr, npv, acc, f1]]


def get_abcd(predict, truth):
    # pos > 0, neg == 0
    n = len(predict)
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(n):
        if predict[i] > 0 and truth[i] > 0:
            tp += 1
        elif predict[i] > 0 and truth[i] == 0:
            fp += 1
        elif predict[i] == 0 and truth[i] == 0:
            tn += 1
        elif predict[i] == 0 and truth[i] > 0:
            fn += 1
    return tp, fp, tn, fn


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
    pre, rec, spec, fpr, npv, acc, f1 = get_performance(tp, fp, tn, fn)
    return [tp, fp, tn, fn, pre, rec, spec, fpr, npv, acc, f1]


def plot_compare(name, fft1, fft2):
    # plot ROC
    fig, ax = plt.subplots()
    ax.set_title('FFT Comparison |  Data: ' + name)
    ax.set_xlabel("False Alarm Rates")
    ax.set_ylabel("Recall")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    # plot diagonal
    x, y = [0.001 * i for i in range(1000)], [0.001 * i for i in range(1000)]
    ax.scatter(x, y, s=4)

    # plot fft peformances
    tmp = {"Accuracy": 0, "Dist2Heaven": 1, "Gini": 2, "InfoGain": 3}
    markers = ['*', 'o']
    colors = ['#800000', '#6B8E23', '#65ff00', '#ff0000']
    for fft in [fft1, fft2]:
        roc = fft.roc
        s_id = fft.best
        k = tmp[fft.criteria]
        for i in range(fft.tree_cnt):
            if i == s_id:
                continue
            ax.scatter(roc[i][0], roc[i][1], marker=markers[k], c=colors[k], s=400-k*300)
            ax.annotate(i, (roc[i][0], roc[i][1]))
        t = 0 if s_id != 0 else 1
        ax.scatter(roc[t][0], roc[t][1], c=colors[k], marker=markers[k], s=400-k*300, label="FFT(" + fft.criteria + ")")

        # plot the best fft
        ax.scatter(roc[s_id][0], roc[s_id][1], c=colors[-k-1], \
               marker=markers[k], s=400-k*300, label="Best_FFT(" + fft.criteria + ")")
        ax.annotate("B_FFT(" + fft.criteria[0] + ")", (roc[s_id][0], roc[s_id][1]))

    # # plot the best fft in red
    # ax.scatter(fft1.roc[fft1.best][0], fft1.roc[fft1.best][1], c='m', \
    #            marker=markers[0], s=100, label="Best_FFT(" + fft1.criteria + ")")
    # ax.scatter(fft2.roc[fft2.best][0], fft2.roc[fft2.best][1], c='#00FF00', \
    #            marker=markers[1], s=100, label="Best_FFT(" + fft2.criteria + ")")
    #
    # ax.annotate("B_FFT(" + fft1.criteria[0] + ")", (fft1.roc[fft1.best][0], fft1.roc[fft1.best][1]))
    # ax.annotate("B_FFT(" + fft2.criteria[0] + ")", (fft2.roc[fft2.best][0], fft2.roc[fft2.best][1]))

    legend = ax.legend(loc='lower right', shadow=True, fontsize='small')
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#CEE5DD')
    # plt.show()

    img_path = os.path.join(data_path, "Compare_" + name + ".png")
    plt.savefig(img_path)


def plot_effort(fft):
    data = fft.test
    SL, NB, EM, SMO = fft.soa_learners
    ours = None



cwd = os.getcwd()
data_path = os.path.join(cwd, "data")
data = {"@ivy":     ["ivy-1.1.csv", "ivy-1.4.csv", "ivy-2.0.csv"],\
        "@lucene":  ["lucene-2.0.csv", "lucene-2.2.csv", "lucene-2.4.csv"],\
        "@poi":     ["poi-1.5.csv", "poi-2.0.csv", "poi-2.5.csv", "poi-3.0.csv"],\
        "@synapse": ["synapse-1.0.csv", "synapse-1.1.csv", "synapse-1.2.csv"],\
        "@velocity":["velocity-1.4.csv", "velocity-1.5.csv", "velocity-1.6.csv"], \
        "@camel": ["camel-1.0.csv", "camel-1.2.csv", "camel-1.4.csv", "camel-1.6.csv"], \
        "@jedit": ["jedit-3.2.csv", "jedit-4.0.csv", "jedit-4.1.csv", "jedit-4.2.csv", "jedit-4.3.csv"], \
        "@log4j": ["log4j-1.0.csv", "log4j-1.1.csv", "log4j-1.2.csv"], \
        "@xalan": ["xalan-2.4.csv", "xalan-2.5.csv", "xalan-2.6.csv", "xalan-2.7.csv"], \
        "@xerces": ["xerces-1.2.csv", "xerces-1.3.csv", "xerces-1.4.csv"]
        }

all_data_filepath = os.path.join(data_path, "dist2h_data.pkl")
if os.path.exists(all_data_filepath):
    all_data = load_obj(all_data_filepath)
else:
    all_data = {}
for name, files in data.iteritems():
    if name not in all_data:
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        print name
        paths = [os.path.join(data_path, file_name) for file_name in files]
        train_df = pd.concat([pd.read_csv(path) for path in paths[:-1]])
        test_df = pd.read_csv(paths[-1])
        train_df['bug'] = train_df['bug'].apply(lambda x: 0 if x == 0 else 1)
        test_df['bug'] = test_df['bug'].apply(lambda x: 0 if x == 0 else 1)
        print "training on: " + ', '.join(files[:-1])
        print "testing on: " + files[-1]
        all_data[name] = {}

        # criterias = ["Dist2Heaven", "Accuracy", "Gini", "InfoGain"]
        criterias = ["Dist2Heaven"] #, "Accuracy", "Gini", "InfoGain"]
        for criteria in criterias:
            print "...................... " + criteria + " ......................"
            fft = FFT()
            fft.criteria = criteria
            fft.title = name
            fft.img_path = os.path.join(data_path, fft.criteria + "_" + name + ".png")
            fft.train, fft.test = train_df, test_df
            fft.run_all()
            all_data[name][criteria] = fft
    # plot_compare(name, all_data[name]["Accuracy"], all_data[name]["Dist2Heaven"])
    # plot_effort(name, all_data[name]["Dist2Heaven"])

if not os.path.exists(all_data_filepath):
    save_obj(all_data, all_data_filepath)
