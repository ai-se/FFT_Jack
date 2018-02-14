import os
import collections
import pandas as pd
import cPickle
import csv

from helpers import load_obj, save_obj
from cross_validation import cross_val

cwd = os.getcwd()
data_path = os.path.join(cwd, "data", "smell")
data = {"DataClass":     ["DataClass.csv"],\
        "FeatureEnvy":  ["FeatureEnvy.csv"],\
        "GodClass":     ["GodClass.csv"],\
        "LongMethod": ["LongMethod.csv"]
        }
target = "SMELLS"


details_path = os.path.join(data_path, 'smell_details_38-MDLP.pkl')
csv_path = os.path.join(data_path, 'smell_details_38-MDLP.csv')
if os.path.exists(details_path):
    performances = load_obj(details_path)
else:
    performances = collections.defaultdict(dict)

p_opt_stat = []
all_performances = {}
classifiers = ["DT", "RF", "LR", "kNN", "FFT-Dist2Heaven"]

for name, file in data.iteritems():
    if name not in performances:
        print name + "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        file_path = os.path.join(data_path, file[0])
        df = pd.read_csv(file_path)
        df[target] = df[target].apply(lambda x: 1 if x else 0)
        for i, clf in enumerate(classifiers):
            print clf + "````````````````````````````````````"
            performances[name][clf] = cross_val(clf=clf, data=df.drop(columns=[target]), label=df[target],
                                              target_label=1, iterations=8, folds=3, title=' + '.join([name, clf]))
    else:
        csv_path = os.path.join(data_path, name + "_performance.csv")
        tmp = pd.DataFrame(performances[name])
        tmp.to_csv(csv_path)

if not os.path.exists(details_path):
    save_obj(performances, details_path)

print 'done'
