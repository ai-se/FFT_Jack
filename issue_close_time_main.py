import os
import collections
import pandas as pd
import cPickle

from helpers import load_obj, save_obj
from new_fft import FFT
from SOA import SOA
from plot import plotROC, plotLOC, plot_compare
from cross_validation import cross_val


folders = ["1 day"] + map(lambda x: str(x) + " days", [7, 14, 30, 90, 180, 365])
classifiers = ["DT", "RF", "LR", "kNN", "FFT-Accuracy", "FFT-Dist2Heaven"]
target = "timeOpen"

cwd = os.getcwd()
data_path = os.path.join(cwd, "data", "issue_close_time")
details_path = os.path.join(data_path, 'issue_close_time_details.pkl')
if os.path.exists(details_path):
    performances = load_obj(details_path)
else:
    performances = {}

for folder in folders:
    if folder not in performances:
        performances[folder] = collections.defaultdict(dict)
        folder_path = os.path.join(data_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                print file + "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path)
                df[target] = df[target].apply(lambda x: 1 if x else 0)
                for i, clf in enumerate(classifiers):
                    print clf + "````````````````````````````````````"
                    performances[folder][file][clf] = cross_val(clf=clf, data=df.drop(columns=[target]), label=df[target],
                                                        target_label=1, iterations=10, folds=10, title=' + '.join([folder, file, clf]))

if not os.path.exists(details_path):
    save_obj(performances, details_path)

print 'done'
