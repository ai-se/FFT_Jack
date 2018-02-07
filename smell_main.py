import os
import collections
import pandas as pd
import cPickle

from helpers import load_obj, save_obj
from new_fft import FFT
from SOA import SOA
from plot import plotROC, plotLOC, plot_compare
from cross_validation import cross_val

cwd = os.getcwd()
data_path = os.path.join(cwd, "data", "smell")
data = {"DataClass":     ["DataClass.csv"],\
        "FeatureEnvy":  ["FeatureEnvy.csv"],\
        "GodClass":     ["GodClass.csv"],\
        "LongMethod": ["LongMethod.csv"]
        }
criterias = ["Accuracy", "Dist2Heaven", "LOC_AUC"] # "Gini", "InfoGain"]
all_data_filepath = os.path.join(data_path, "smells.pkl")
target = "SMELLS"

if os.path.exists(all_data_filepath):
    all_data = load_obj(all_data_filepath)
else:
    all_data = {}

p_opt_stat = []
cnts = [collections.defaultdict(int) for _ in xrange(len(criterias))]
all_performances = {}
classifiers = {"DT", "RF", "LR", "kNN"}

performances = collections.defaultdict(dict)
for name, file in data.iteritems():
    if name not in all_data:
        print name + "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        file_path = os.path.join(data_path, file[0])
        df = pd.read_csv(file_path)
        df[target] = df[target].apply(lambda x: 1 if x else 0)
        for i, clf in enumerate(classifiers):
            print clf + "````````````````````````````````````"
            performances[name][clf] = cross_val(clf=clf, data=df.drop(columns=[target]), label=df[target],
                                              target_label=1, folds=10, title=' + '.join([name, clf]))

print 'done'
details_path = os.path.join(data_path, 'performance_details_SOA.p')
cPickle.dump(performances, open(details_path, 'wb'))

