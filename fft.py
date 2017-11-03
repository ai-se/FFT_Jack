import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC
import collections
import os

from ABCD import ABCD

PRE, REC, SPEC, ACC, F1 = 5, 4, 3, 2, 1
# training / testing split
class FFT(object):
    def __init__(self, df=None, max_level=4, goal=PRE):
        cnt = 2 ** max_level
        self.df = df
        self.max_level = max_level
        self.tree_cnt = cnt
        self.tree_depth = [0] * cnt
        self.target = "bug"
        self.goal_chase = -goal
        self.best = -1
        self.ignore = {"name", "version", 'name.1'}
        self.train, self.test = self.split_data(df)
        self.trees = [{}] * cnt
        self.tree_plotted = [False] * cnt
        self.performance = [collections.defaultdict(dict) for _ in xrange(cnt)]
        self.selected = [{} for _ in range(cnt)]
        self.tree_info = [collections.defaultdict(dict) for _ in xrange(cnt)]
        self.soa = None


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
        pre, rec, spec, acc, f1 = get_performance(tp, fp, tn, fn)
        return tp, fp, tn, fn, pre, rec, spec, acc, f1


    "Given tree and level, get the node info"
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
        tp, fp, tn, fn = self.performance[t_id][i][(c, d, t, r)][:4]
        if (r and not reverse) or (r == 0 and reverse):
            cues_used = 1.0 * (i + 1) * (tp + fp) / self.df.shape[0]
            self.tree_info[t_id]["general"] = [x + y for x, y in zip(self.tree_info[t_id]["general"], [tp, fp, 0, 0, cues_used])]
            metric = "\tFalse Alarm: " + str(fp) + ", Hit: " + str(tp)
        else:
            cues_used = 1.0 * (i + 1) * (tn + fn) / self.df.shape[0]
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
        roc = [None] * self.tree_cnt
        print "\n-------------------------------------------------------"
        print "ID\t MCU\t PRE\t REC\t SPEC\t ACC\t F1"
        goal = self.goal_chase
        selected = None
        for i in range(self.tree_cnt):
            self.describe_tree(i)
            tp, fp, tn, fn, mcu = self.tree_info[i]["general"]
            metric = get_performance(tp, fp, tn, fn)
            roc[i] = [1-metric[-SPEC], metric[-REC]]
            if not selected or metric[goal] > selected[goal]:
                selected = [i] + metric
            print " \t ".join([str(x) for x in [i, round(mcu,1)] + metric])
        print "-------------------------------------------------------"
        print "The selected tree id is :" + str(selected[0])

        # Get the state of the art classifiers.
        if not self.soa:
            self.get_soa()


        # plot ROC
        fig, ax = plt.subplots()
        ax.set_title('ROC')
        ax.set_xlabel("False Alarm Rates")
        ax.set_ylabel("Recall")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        # plot diagonal
        x, y = [0.001 * i for i in range(1000)], [0.001 * i for i in range(1000)]
        ax.scatter(x, y, s=4)
        # plot fft peformances
        ax.scatter(x[selected[0]], y[selected[0]], c='r', s=100)
        ax.annotate("B_FFT", (x[selected[0]], y[selected[0]]))
        for i in range(self.tree_cnt):
            ax.scatter(roc[i][0], roc[i][1], s=100)
            ax.annotate(i, (roc[i][0], roc[i][1]))
        # plot state of the art performance
        soa_color = ['g', 'c', 'm', 'y']
        soa_name = ["SL", "NB", "EM", "SMO"]
        for i in range(4):
            ax.scatter(1-self.soa[i][-3], self.soa[i][-4], c=soa_color[i], s=100)
            ax.annotate(soa_name[i], (1-self.soa[i][-3], self.soa[i][-4]))
        plt.show()
        return selected[0]


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
        tp, fp, tn, fn, mcu = self.tree_info[t_id]["general"]
        print "======================================="
        print "\t".join([x.ljust(6, " ") for x in ["TP", "FP", "TN", "FN"]])
        print "\t".join([str(x).ljust(6, " ") for x in [tp, fp, tn, fn]])
        print "\t".join([str(x).ljust(6, " ") for x in ["MCU", "PRE", "REC", "SPEC", "ACC", "F1"]])
        print "\t".join([str(x).ljust(6, " ") for x in [round(mcu,1)] + get_performance(tp, fp, tn, fn)])
        print "======================================="


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
            if cue in self.ignore or cue == self.target:
                continue
            threshold = data[cue].median()
            for direction in "><":
                metrics = self.get_metrics(data, cue, direction, threshold, decision)
                self.performance[t_id][level][(cue, direction, threshold, decision)] = metrics
                goal = self.goal_chase
                if not cur_selected or metrics[goal] > self.performance[t_id][level][cur_selected][goal]:
                    cur_selected = (cue, direction, threshold, decision)
        self.selected[t_id][level] = cur_selected
        s_cue, s_dirc, s_thre, s_decision = cur_selected
        undecided = data.loc[data[s_cue] <= s_thre] if s_dirc == ">" else data.loc[data[s_cue] >= s_thre]
        self.grow(undecided, t_id, level + 1)


    "Build all possible tress."
    def build_trees(self):
        self.structures = self.get_all_structure()
        for i in range(len(self.structures)):
            self.grow(self.train, i, 0)


    "Get performance of state of the art classifiers"
    def get_soa(self):
        SL = LogisticRegression()   # simple logistic
        NB = GaussianNB()           # Naive Bayes\
        EM = GaussianMixture()      # Expectation Maximization
        SMO = LinearSVC()           # Support Vector Machines
        train_data, train_label = self.train.iloc[:, 3:-1], self.train.iloc[:, -1]
        test_data, test_label = self.test.iloc[:, 3:-1], self.test.iloc[:, -1]
        m_SL = do_classification(train_data, test_data, train_label, test_label, SL)
        m_NB = do_classification(train_data, test_data, train_label, test_label, NB)
        m_EM = do_classification(train_data, test_data, train_label, test_label, EM)
        m_SMO = do_classification(train_data, test_data, train_label, test_label, SMO)
        self.soa = [m_SL, m_NB, m_EM, m_SMO]



"Given TP, FP, TN, FN, get all the other metrics. "
def get_performance(tp, fp, tn, fn):
    pre = 1.0 * tp / (tp + fp) if (tp + fp) != 0 else 0
    rec = 1.0 * tp / (tp + fn) if (tp + fn) != 0 else 0
    spec = 1.0 * tn / (tn + fp) if (tn + fp) != 0 else 0
    acc = 1.0 * (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) != 0 else 0
    return [round(x, 1) for x in [pre, rec, spec, acc, f1]]


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


def do_classification(train_data, test_data, train_label, test_label, clf=''):
    if not clf:
        clf = LinearSVC()
    clf.fit(train_data, train_label)
    prediction = clf.predict(test_data)
    tp, fp, tn, fn = get_abcd(prediction, np.array(test_label))
    pre, rec, spec, acc, f1 = get_performance(tp, fp, tn, fn)
    return tp, fp, tn, fn, pre, rec, spec, acc, f1


cwd = os.getcwd()
csv_path = os.path.join(cwd, "ivy-2.0.csv")
df = pd.read_csv(csv_path)
print df.describe()

fft = FFT(df, goal=PRE)
fft.get_soa()
fft.build_trees()
fft.plot_tree(show_metrics=True)
# fft.grow(fft.train)
# fft.describe_tree(show_metrics=True)

print "done"
