import collections

from helpers import get_performance, get_score

PRE, REC, SPEC, FPR, NPV, ACC, F1 = 7, 6, 5, 4, 3, 2, 1
MATRIX = "\t".join(["\tTP", "FP", "TN", "FN"])
PERFORMANCE = " \t".join(["\tCLF", "PRE ", "REC", "SPE", "FPR", "NPV", "ACC", "F_1"])


class FFT(object):
    def __init__(self, max_level=1):
        self.max_depth = max_level - 1
        cnt = 2 ** self.max_depth
        self.tree_cnt = cnt
        self.tree_depths = [0] * cnt
        self.best = -1

        self.target = "bug"
        self.ignore = {"name", "version", 'name.1'}
        self.criteria = "Dist2Heaven"

        self.data_name = ''
        self.train, self.test = None, None

        self.structures = None
        self.computed_cache = {}

        self.selected = [{} for _ in range(cnt)]
        self.tree_scores = [None] * cnt
        self.node_descriptions = [None] * cnt
        self.performance_on_train = [collections.defaultdict(dict) for _ in xrange(cnt)]
        self.performance_on_test = [None] * cnt

    "Build all possible tress."

    def build_trees(self):
        self.structures = self.get_all_structure()
        for i in range(self.tree_cnt):
            self.grow(self.train, i, 0, [0, 0, 0, 0])

    "Evaluate all tress built."

    def eval_trees(self):
        for i in range(self.tree_cnt):
            self.eval_tree(i)

    "Find the best tree based on the score."

    def find_best_tree(self):
        if self.tree_scores and self.tree_scores[0]:
            return
        if not self.performance_on_test or not self.performance_on_test[0]:
            self.eval_trees()
        print "\t----- PERFORMANCES FOR ALL FFTs -----"
        print PERFORMANCE + " \t" + self.criteria
        best = [-1, float('inf')]
        for i in range(self.tree_cnt):
            all_metrics = self.performance_on_test[i]
            score = get_score(self.criteria, all_metrics[:4])
            self.tree_scores[i] = score
            if score < best[-1]:
                best = [i, score]
            print "\t" + "\t".join(
                ["FFT(" + str(i) + ")"] + \
                [str(x).ljust(5, "0") for x in all_metrics[4:] + [score]])
        print "\tThe best tree found is: FFT(" + str(best[0]) + ")"
        self.best = best[0]
        self.print_tree(best[0])

    "Given how the decision is made, get the description for the node."

    def describe_decision(self, t_id, level, metrics, reversed=False):
        cue, direction, threshold, decision = self.selected[t_id][level]
        tp, fp, tn, fn = metrics
        results = ["\'Good\'", "\'Bug!\'"]
        description = ("\t| " * (level + 1) + \
                       " ".join([cue, direction, str(threshold)]) + \
                       "\t--> " + results[1 - decision if reversed else decision]).ljust(30, " ")
        pos = "\tFalse Alarm: " + str(fp) + ", Hit: " + str(tp)
        neg = "\tCorrect Rej: " + str(tn) + ", Miss: " + str(fn)
        if not reversed:
            description += pos if decision == 1 else neg
        else:
            description += neg if decision == 1 else pos
        return description

    "Given how the decision is made, get the performance for this decision."

    def eval_decision(self, data, cue, direction, threshold, decision):
        if direction == ">":
            pos, neg = data.loc[data[cue] > threshold], data.loc[data[cue] <= threshold]
        else:
            pos, neg = data.loc[data[cue] < threshold], data.loc[data[cue] >= threshold]
        if decision == 1:
            undecided = neg
        else:
            pos, neg = neg, pos
            undecided = pos

        tp = len(pos.loc[pos[self.target] == 1])
        fp = len(pos.loc[pos[self.target] == 0])
        tn = len(neg.loc[neg[self.target] == 0])
        fn = len(neg.loc[neg[self.target] == 1])
        # pre, rec, spec, fpr, npv, acc, f1 = get_performance([tp, fp, tn, fn])
        # return undecided, [tp, fp, tn, fn, pre, rec, spec, fpr, npv, acc, f1]
        return undecided, [tp, fp, tn, fn]

    "Evaluate the performance of the given tree on the test data."

    def eval_tree(self, t_id):
        if self.performance_on_test[t_id]:
            return
        depth = self.tree_depths[t_id]
        self.node_descriptions[t_id] = [[] for _ in range(depth + 1)]
        TP, FP, TN, FN = 0, 0, 0, 0
        data = self.test
        for level in range(depth + 1):
            cue, direction, threshold, decision = self.selected[t_id][level]
            undecided, metrics = self.eval_decision(data, cue, direction, threshold, decision)
            tp, fp, tn, fn = self.update_metrics(level, depth, decision, metrics)
            description = self.describe_decision(t_id, level, metrics)
            self.node_descriptions[t_id][level] += [description]
            TP, FP, TN, FN = TP + tp, FP + fp, TN + tn, FN + fn
            if len(undecided) == 0:
                break
            data = undecided
        description = self.describe_decision(t_id, level, metrics, reversed=True)
        self.node_descriptions[t_id][level] += [description]

        pre, rec, spec, fpr, npv, acc, f1 = get_performance([TP, FP, TN, FN])
        self.performance_on_test[t_id] = [TP, FP, TN, FN, pre, rec, spec, fpr, npv, acc, f1]

    "Grow the t_id_th tree for the level with the given data"

    def grow(self, data, t_id, level, cur_performance):
        """
        :param data: current data for future tree growth
        :param t_id: tree id
        :param level: level id
        :return: None
        """
        if level > self.max_depth:
            return
        if len(data) == 0:
            print "?????????????????????? Early Ends ???????????????????????"
            return
        self.tree_depths[t_id] = level
        decision = self.structures[t_id][level]
        structure = tuple(self.structures[t_id][:level + 1])
        cur_selected = self.computed_cache.get(structure, None)
        TP, FP, TN, FN = cur_performance
        if not cur_selected:
            for cue in list(data):
                if cue in self.ignore or cue == self.target:
                    continue
                threshold = data[cue].median()
                for direction in "><":
                    undecided, metrics = self.eval_decision(data, cue, direction, threshold, decision)
                    tp, fp, tn, fn = self.update_metrics(level, self.max_depth, decision, metrics)
                    score = get_score(self.criteria, [TP + tp, FP + fp, TN + tn, FN + fn])
                    # score = get_score(self.criteria, metrics)
                    # if not cur_selected or metrics[goal] > self.performance_on_train[t_id][level][cur_selected][goal]:
                    if not cur_selected or score < cur_selected['score']:
                        cur_selected = {'rule': (cue, direction, threshold, decision), \
                                        'undecided': undecided, \
                                        'metrics': [TP + tp, FP + fp, TN + tn, FN + fn], \
                                        # 'metrics': metrics,
                                        'score': score}
            self.computed_cache[structure] = cur_selected
        self.selected[t_id][level] = cur_selected['rule']
        self.performance_on_train[t_id][level] = cur_selected['metrics'] + get_performance(cur_selected['metrics'])
        self.grow(cur_selected['undecided'], t_id, level + 1, cur_selected['metrics'])

    "Given tree id, print the specific tree and its performances."

    def print_tree(self, t_id):
        depth = self.tree_depths[t_id]
        for i in range(depth + 1):
            print self.node_descriptions[t_id][i][0]
        print self.node_descriptions[t_id][i][1]

        print "\t----- CONFUSION MATRIX -----"
        print MATRIX
        print "\t" + "\t".join(map(str, self.performance_on_test[t_id][:4]))

        print "\t----- PERFORMANCES ON TEST DATA -----"
        print PERFORMANCE + " \t" + "Dist2Heaven"
        dist2heaven = get_score("Dist2Heaven", self.performance_on_test[t_id][:4])
        print "\t" + "\t".join(
            ["FFT(" + str(self.best) + ")"] + \
            [str(x).ljust(5, "0") for x in self.performance_on_test[t_id][4:] + [dist2heaven]])
            # map(str, ["FFT(" + str(self.best) + ")"] + self.performance_on_test[t_id][4:] + [dist2heaven]))

    "Get all possible tree structure"

    def get_all_structure(self):
        def dfs(cur, n):
            if len(cur) == n:
                ans.append(cur)
                return
            dfs(cur + [1], n)
            dfs(cur + [0], n)

        if self.max_depth < 0:
            return []
        ans = []
        dfs([], self.max_depth + 1)
        return ans[:self.tree_cnt]

    "Update the metrics(TP, FP, TN, FN) based on the decision."

    def update_metrics(self, level, depth, decision, metrics):
        tp, fp, tn, fn = metrics
        if level < depth:  # Except the last level, only part of the data(pos or neg) is decided.
            if decision == 1:
                tn, fn = 0, 0
            else:
                tp, fp = 0, 0
        return tp, fp, tn, fn