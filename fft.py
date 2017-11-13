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
SEED = 666
PRE, REC, SPEC, FPR, NPV, ACC, F1 = 7, 6, 5, 4, 3, 2, 1
COLORS = ["#800000", "#6B8E23", "#0000CD", "#8A2BE2", "#FFFF00", "#00FF00", "#00FFFF", "#FF00FF"]
# training / testing split
class FFT(object):
    def __init__(self, df=None, max_level=4, goal=REC):
        cnt = 2 ** max_level
        self.df = df
        self.title = ""
        self.img_path = ""
        #self.train, self.test = self.split_data(df)
        self.max_level = max_level
        self.tree_cnt = cnt
        self.tree_depth = [0] * cnt
        self.target = "bug"
        self.criteria = "Dist2Heave"
        self.goal_chase = -goal
        self.best = -1
        self.ignore = {"name", "version", 'name.1'}
        self.trees = [{}] * cnt
        self.tree_plotted = [False] * cnt
        self.performance = [collections.defaultdict(dict) for _ in xrange(cnt)]
        self.selected = [{} for _ in range(cnt)]
        self.tree_info = [collections.defaultdict(dict) for _ in xrange(cnt)]
        self.soa = None
        self.soa_name = ["SL", "NB", "EM", "SMO"]


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
        roc = [None] * self.tree_cnt
        print "#### Performance for all FFT generated. ####"
        print "-------------------------------------------------------"
        print "ID\t MCU\t PRE\t REC\t SPEC\t NPV\t ACC\t F1"
        goal = self.goal_chase
        selected = None
        for i in range(self.tree_cnt):
            self.describe_tree(i)
            tp, fp, tn, fn, mcu = self.tree_info[i]["general"]
            metric = get_performance(tp, fp, tn, fn)
            roc[i] = [metric[-FPR], metric[-REC]]
            dist2heaven = roc[i][0]**2 + (1-roc[i][1])**2
            if not selected or dist2heaven < selected[1]:
                selected = [i] + [dist2heaven]
            print " \t ".join([str(x) for x in [i, round(mcu,1)] + metric])
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
        tmp = {"Accuracy":0, "Dist2Heave": 1, "Gini": 2, "InfoGain": 3}
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
        ax.scatter(roc[s_id][0], roc[s_id][1], c='r', s=100, label="Best_FFT")
        ax.annotate("B_FFT", (roc[s_id][0], roc[s_id][1]))

        # plot state of the art performance
        for i in range(4):
            ax.scatter(self.soa[i][-FPR], self.soa[i][-REC], c=COLORS[i+4], s=100, label=self.soa_name[i])
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
        tp, fp, tn, fn, mcu = self.tree_info[t_id]["general"]

        print "\n#### Performance of the best FFT ####"
        print "======================================="
        print "\t".join([x.ljust(6, " ") for x in ["TP", "FP", "TN", "FN"]])
        print "\t".join([str(x).ljust(6, " ") for x in [tp, fp, tn, fn]])
        print "\t".join([str(x).ljust(6, " ") for x in ["MCU", "PRE", "REC", "SPEC", "FPR", "NPV", "ACC", "F1"]])
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
            if cue in self.ignore or cue == self.target:
                continue
            threshold = data[cue].median()
            for direction in "><":
                metrics = self.get_metrics(data, cue, direction, threshold, decision)
                self.performance[t_id][level][(cue, direction, threshold, decision)] = metrics
                goal = self.goal_chase
                if self.criteria == "Accuracy":
                    score = -metrics[-ACC]
                elif self.criteria == "Dist2Heave":
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
                # if not cur_selected or metrics[goal] > self.performance[t_id][level][cur_selected][goal]:
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


    "Get performance of state of the art classifiers"
    def get_soa(self):
        SL = LogisticRegression(random_state=SEED)   # simple logistic
        NB = GaussianNB()           # Naive Bayes\
        EM = GaussianMixture(random_state=SEED, init_params='kmeans', n_components=2)      # Expectation Maximization
        SMO = LinearSVC(random_state=SEED)           # Support Vector Machines
        train_data, train_label = self.train.iloc[:, 3:-1], self.train.iloc[:, -1]
        test_data, test_label = self.test.iloc[:, 3:-1], self.test.iloc[:, -1]
        m_SL = do_classification(train_data, test_data, train_label, test_label, SL)
        m_NB = do_classification(train_data, test_data, train_label, test_label, NB)
        m_EM = do_classification(train_data, test_data, train_label, test_label, EM)
        m_SMO = do_classification(train_data, test_data, train_label, test_label, SMO)
        self.soa = [m_SL, m_NB, m_EM, m_SMO]


    def run_all(self):
        self.get_soa()
        self.build_trees()
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


def do_classification(train_data, test_data, train_label, test_label, clf=''):
    if not clf:
        clf = LinearSVC()
    clf.fit(train_data, train_label)
    prediction = clf.predict(test_data)
    tp, fp, tn, fn = get_abcd(prediction, np.array(test_label))
    pre, rec, spec, fpr, npv, acc, f1 = get_performance(tp, fp, tn, fn)
    return [tp, fp, tn, fn, pre, rec, spec, fpr, npv, acc, f1]


cwd = os.getcwd()
data_path = os.path.join(cwd, "data")
data = {"@ivy":     ["ivy-1.1.csv", "ivy-1.4.csv", "ivy-2.0.csv"],\
        "@lucene":  ["lucene-2.0.csv", "lucene-2.2.csv", "lucene-2.4.csv"],\
        "@poi":     ["poi-1.5.csv", "poi-2.0.csv", "poi-2.5.csv", "poi-3.0.csv"],\
        "@synapse": ["synapse-1.0.csv", "synapse-1.1.csv", "synapse-1.2.csv"],\
        "@velocity":["velocity-1.4.csv", "velocity-1.5.csv", "velocity-1.6.csv"]}


for name, files in data.iteritems():
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print name
    paths = [os.path.join(data_path, file_name) for file_name in files]
    train_df = pd.concat([pd.read_csv(path) for path in paths[:-1]])
    test_df = pd.read_csv(paths[-1])
    train_df['bug'] = train_df['bug'].apply(lambda x: 0 if x == 0 else 1)
    test_df['bug'] = test_df['bug'].apply(lambda x: 0 if x == 0 else 1)
    print "training on: " + ', '.join(files[:-1])
    print "testing on: " + files[-1]

    for criteria in ["Dist2Heave", "Accuracy", "Gini", "InfoGain"]:
        print "...................... " + criteria + " ......................"
        fft = FFT()
        fft.criteria = criteria
        fft.title = name
        fft.img_path = os.path.join(data_path, fft.criteria + "_" + name + ".png")
        fft.train, fft.test = train_df, test_df
        fft.run_all()


