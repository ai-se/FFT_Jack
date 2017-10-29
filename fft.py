import pandas as pd
import numpy as np
import collections
import os

PRE, REC, SPEC, ACC, F1 = 0, 1, 2, 3, 4
# training / testing split
class FFT(object):
    def __init__(self, df=None):
        self.df = df
        self.max_level = 4
        self.cur_level = 0
        self.last_cur = ""
        self.root = None
        self.target = "bug"
        self.goal_chase = F1 # -2 --> acc
        self.ignore = {"name", "version", 'name.1'}
        self.train, self.test = self.split_data(df)
        self.performance = collections.defaultdict(dict)
        self.decision = {}
        self.tree_info = collections.defaultdict(dict)
        self.tree_plotted = False


    def split_data(self, df):
        mask = np.random.rand(len(df)) <= 0.6
        train, test = df[mask], df[~mask]
        return train, test


    def get_metrics(self, data, cue, direction, threshold):
        if direction == ">":
            pos, neg = data.loc[data[cue] > threshold], data.loc[data[cue] <= threshold]
        else:
            pos, neg = data.loc[data[cue] < threshold], data.loc[data[cue] >= threshold]
        # reverse pos and neg every 2 level
        if self.cur_level % 2 == 1:
            pos, neg = neg, pos

        tp = len(pos.loc[pos[self.target] == 1])
        fp = len(pos.loc[pos[self.target] == 0])
        tn = len(neg.loc[neg[self.target] == 0])
        fn = len(neg.loc[neg[self.target] == 1])
        pre, rec, spec, acc, f1 = self.get_performance(tp, fp, tn, fn)
        return tp, fp, tn, fn, pre, rec, spec, acc, f1


    def get_performance(self, tp, fp, tn, fn):
        pre = 1.0 * tp / (tp + fp) if (tp + fp) != 0 else 0
        rec = 1.0 * tp / (tp + fn) if (tp + fn) != 0 else 0
        spec = 1.0 * tn / (tn + fp) if (tn + fp) != 0 else 0
        acc = 1.0 * (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        f1 = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) != 0 else 0
        return pre, rec, spec, acc, f1


    def get_node_info(self, i, reverse=False):
        # cue, direction, threshold
        c, d, t = self.decision[i]
        if reverse:
            d = ">" if d == "<" else "<"
        tp, fp, tn, fn = self.performance[i][(c, d, t)][:4]
        results = ["[Bug]", "[No!]"]

        node = "| " * i + " ".join([c, d, str(t)]) + "\t--> " + results[~(i%2) if reverse else i%2]
        self.tree_info[i]["description"] = self.tree_info[i].get("description", []) + [node]

        if "general" not in self.tree_info:
            self.tree_info["general"] = [0] * 5 # TP, FP, TN, FN, MCU

        if (i % 2 and not reverse) or (i % 2 == 0 and reverse):
            cues_used = 1.0 * (i + 1) * (tn + fn) / self.df.shape[0]
            self.tree_info["general"] = [x + y for x, y in zip(self.tree_info["general"], [0, 0, tn, fn, cues_used])]
            metric = "\tCorrect Rej: " + str(tn) + ", Miss: " + str(fn)

        else:
            cues_used = 1.0 * (i + 1) * (tp + fp) / self.df.shape[0]
            self.tree_info["general"] = [x + y for x, y in zip(self.tree_info["general"], [tp, fp, 0, 0, cues_used])]
            metric = "\tFalse Alarm: " + str(fp) + ", Hit: " + str(tp)
        self.tree_info[i]["metric"] = self.tree_info[i].get("metric", []) + [metric]



    def get_tree_performance(self):
        n = self.cur_level
        row_cnt = self.df.shape[0]
        mcu, pci, sens, spec, acc = 0, 0, 0, 0, 0
        for i in range(n):
            self.get_node_info(i)
            if i == n - 1:
                self.get_node_info(i, reverse=True)


    def get_tree(self, show_metrics=False):
        if not self.tree_plotted:
            self.tree_plotted = True
            self.get_tree_performance()

        n = self.cur_level
        for i in range(n):
            print self.tree_info[i]["description"][0] + \
                  " \t" * (n-i) + (self.tree_info[i]["metric"][0] if show_metrics else "")
        print self.tree_info[i]["description"][1] + \
              " \t" * (n-i) + (self.tree_info[i]["metric"][1] if show_metrics else "")
        print self.tree_info["general"]
        tp, fp, tn, fn, mcu = self.tree_info["general"]
        print self.get_performance(tp, fp, tn, fn)


    def grow(self, data):
        level = self.cur_level
        if level >= self.max_level:
            return

        # split on meadian and get all performance scores.
        selected = None
        for cue in list(data):
            if cue in self.ignore or cue == self.target:
                continue
            threshold = data[cue].median()
            for direction in "><":
                metrics = self.get_metrics(data, cue, direction, threshold)
                self.performance[level][(cue, direction, threshold)] = metrics
                if not selected or metrics[self.goal_chase] > self.performance[level][selected][self.goal_chase]:
                    selected = (cue, direction, threshold)

        self.decision[level] = selected
        s_cue, s_dirc, s_thre = selected
        undecided = data.loc[data[s_cue] <= s_thre] if s_dirc == ">" else data.loc[data[s_cue] >= s_thre]
        if len(undecided) == 0:
            return

        self.cur_level += 1
        self.grow(undecided)


cwd = os.getcwd()
csv_path = os.path.join(cwd, "ivy-2.0.csv")
df = pd.read_csv(csv_path)
print df.describe()
print df["lcom"].median()

fft = FFT(df)
fft.grow(fft.train)
fft.get_tree(show_metrics=True)

print "done"



